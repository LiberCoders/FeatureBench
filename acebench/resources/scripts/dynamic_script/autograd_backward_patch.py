import sys
import traceback
from typing import Any, Callable, Dict, Optional, Tuple


class AutogradBackwardPatcher:
    """Patch torch.autograd.Function.backward to record calls.

    This helper keeps backward-related hook logic outside the main tracing script.
    It relies on shared state from the tracer instance (debug flags, call recorder,
    call stack utilities, etc.).
    """

    def __init__(self, tracer: Any, call_record_cls: Any):
        self.tracer = tracer  # DependencyTracer instance (dynamic_script.py's self), shared state for the patch
        self.call_record_cls = call_record_cls  # CallRecord dataclass
        
        # Track replaced backward methods: key=autograd Function subclass, value=original function
        # We wrap backward on the key class; value is the original backward method.
        self.hooked_backward_methods: Dict[Any, Callable[..., Any]] = {}

    def hook(self) -> None:
        """Install patches for backward methods of loaded autograd Function subclasses."""
        try:
            import torch  # type: ignore
            import gc
            import inspect

            if self.tracer.debug:
                print("\nStarting to hook autograd.Function.backward...")

            # Track how many backward methods were hooked
            hooked_count = 0

            # Iterate over all objects currently loaded in memory
            for obj in gc.get_objects():
                try:
                    # Skip non-class objects
                    if not inspect.isclass(obj):
                        continue
                    # Skip objects that are not torch.autograd.Function subclasses
                    if not issubclass(obj, torch.autograd.Function):
                        continue
                    # Skip the base class itself
                    if obj is torch.autograd.Function:
                        continue
                    # Skip classes without backward
                    if not hasattr(obj, "backward"):
                        continue

                    # Get the backward method
                    backward_method = getattr(obj, "backward")

                    # Skip if already hooked
                    if obj in self.hooked_backward_methods:
                        continue

                    # Record original method
                    self.hooked_backward_methods[obj] = backward_method

                    try:
                        # If the class defines backward itself
                        if "backward" in obj.__dict__:
                            # Get the method
                            raw_backward = obj.__dict__["backward"]
                            if isinstance(raw_backward, staticmethod):
                                actual_func = raw_backward.__func__
                            #
                            elif isinstance(raw_backward, classmethod):
                                actual_func = raw_backward.__func__
                            else:
                                actual_func = raw_backward
                            # Unwrap
                            unwrapped_func = inspect.unwrap(actual_func)
                            # Get file name and line number
                            backward_file = inspect.getfile(unwrapped_func)
                            backward_line = inspect.getsourcelines(unwrapped_func)[1]

                        # backward is inherited from a parent class
                        else:
                            unwrapped_method = inspect.unwrap(backward_method)
                            backward_file = inspect.getfile(unwrapped_method)
                            backward_line = inspect.getsourcelines(unwrapped_method)[1]

                    except Exception:
                        backward_file = obj.__module__ if hasattr(obj, "__module__") else "<unknown>"
                        backward_line = 0

                    # Build wrapper
                    wrapper = self._make_wrapper(
                        original_backward=backward_method,
                        cls_obj=obj,
                        backward_file=backward_file,
                        backward_line=backward_line,
                    )

                    # Replace method
                    if "backward" in obj.__dict__:
                        raw_backward = obj.__dict__["backward"]
                        if isinstance(raw_backward, staticmethod):
                            setattr(obj, "backward", staticmethod(wrapper))
                        elif isinstance(raw_backward, classmethod):
                            setattr(obj, "backward", classmethod(wrapper))
                        else:
                            setattr(obj, "backward", wrapper)
                    else:
                        setattr(obj, "backward", staticmethod(wrapper))

                    hooked_count += 1

                    if self.tracer.debug:
                        print(f"  ✓ Hooked: {obj.__name__}.backward")

                except (TypeError, AttributeError):
                    continue

            if self.tracer.debug:
                print(f"Hooked {hooked_count} autograd.Function subclasses' backward methods\n")

        except ImportError:
            if self.tracer.debug:
                print("PyTorch not detected; skipping autograd backward hook\n")
        except Exception as exc:
            if self.tracer.debug:
                print(f"Error while hooking autograd backward: {exc}\n")
                traceback.print_exc()

    def unhook(self) -> None:
        """Restore patched backward methods."""
        try:
            if self.tracer.debug and self.hooked_backward_methods:
                print(f"\nRestoring {len(self.hooked_backward_methods)} backward methods...")

            for cls, original_backward in self.hooked_backward_methods.items():
                try:
                    setattr(cls, "backward", original_backward)
                except Exception:
                    continue

            self.hooked_backward_methods.clear()
        except Exception as exc:
            if self.tracer.debug:
                print(f"Error while restoring backward methods: {exc}")

    def _make_wrapper(
        self,
        original_backward: Callable[..., Any],  # Original backward method
        cls_obj: Any,   # autograd Function subclass
        backward_file: str, # File path where backward is defined
        backward_line: int, # Line number where backward is defined
    ) -> Callable[..., Any]:

        # DependencyTracer instance (dynamic_script.py's self)
        tracer = self.tracer

        def backward_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Class name
            cls_name = cls_obj.__name__

            # Who called this backward
            caller_file: Optional[str] = None
            caller_name: Optional[str] = None
            caller_line: Optional[int] = None

            # Get info from the main call stack
            call_stack = tracer._get_call_stack()

            for i in range(len(call_stack) - 1, -1, -1):
                stack_file, stack_name, stack_line = call_stack[i]
                # Find the closest caller inside the repo
                if tracer._is_repo_file(stack_file):
                    caller_file = stack_file
                    caller_name = stack_name
                    caller_line = stack_line
                    break

            # Fallback to the top of the stack
            if caller_file is None and call_stack:
                caller_file, caller_name, caller_line = call_stack[-1]

            # If still not found, use inspect to grab the previous frame
            if caller_file is None:
                import inspect

                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_frame = frame.f_back
                    caller_file = caller_frame.f_code.co_filename
                    caller_name = caller_frame.f_code.co_name
                    caller_line = caller_frame.f_lineno
                else:
                    caller_file = "<unknown>"
                    caller_name = "<unknown>"
                    caller_line = 0

            # Record the caller
            record_key = (
                f"{caller_file}:{caller_name}:{caller_line}->"
                f"{backward_file}:backward:{backward_line}"
            )
            if record_key not in tracer.call_records_set:
                record = self.call_record_cls(
                    caller_file=caller_file,
                    caller_name=caller_name,
                    caller_line=caller_line,
                    callee_file=backward_file,
                    callee_name="backward",
                    callee_line=backward_line,
                )
                with tracer.trace_lock:
                    tracer.call_records_set.add(record_key)
                    tracer.call_records.append(record)

                if tracer.debug:
                    print(
                        f"[Hook] backward called: {cls_name}.backward @ {backward_file}:{backward_line}"
                    )

            # Save current profile function
            previous_profile = sys.getprofile()
            # Switch profile function so tracer also captures Python calls inside backward
            sys.setprofile(tracer._trace_calls)

            # Push onto stack
            call_stack.append((backward_file, "backward", backward_line))
            try:
                # After hook logic, call original backward
                return original_backward(*args, **kwargs)
            finally:
                if call_stack:
                    call_stack.pop()
                # Restore original profile
                sys.setprofile(previous_profile)

        # Return wrapper
        return backward_wrapper
