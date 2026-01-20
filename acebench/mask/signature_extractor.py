"""
Signature extractor for functions, classes, and modules in Python source.
"""
import ast
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class FunctionSignature:
    """Function signature information."""
    name: str               # Function name, e.g. 'forward' or '__init__'
    qualified_name: str = ""  # Fully-qualified name, e.g. 'forward' (top-level) or 'ClassName.forward'
    args: List[str] = field(default_factory=list)  # Arg list, e.g. ['self', 'x: int', 'y: str = "default"', '*args', '**kwargs']
    return_annotation: Optional[str] = None # Return annotation, e.g. 'Dict[str, int]' or 'None'
    docstring: Optional[str] = None # Function docstring
    decorators: List[str] = field(default_factory=list) # Decorators, e.g. ['@staticmethod', '@property', '@lru_cache(maxsize=128)']
    is_async: bool = False  # Async function flag (async def)
    nested_functions: List['FunctionSignature'] = field(default_factory=list)  # Nested functions defined in this function
    nested_classes: List['ClassSignature'] = field(default_factory=list)       # Nested classes defined in this function
    line_number: int = 0  # Function definition line (disambiguate same names)
    col_offset: int = 0  # Column offset (indentation)
    full_id: str = ""  # Full identifier: file_path::qualified_name::line_number
    content: Optional[str] = None  # Source snippet


@dataclass
class ClassSignature:
    """Class signature information."""
    name: str                                                       # Class name, e.g. 'MyModel' or 'BaseTransformer'
    qualified_name: str = ""                                        # Fully-qualified name, e.g. 'MyModel' or 'OuterClass.InnerClass'
    bases: List[str] = field(default_factory=list)                  # Base classes, e.g. ['nn.Module', 'ABC']
    nested_functions: List[FunctionSignature] = field(default_factory=list)  # Methods/functions defined in class body
    nested_classes: List['ClassSignature'] = field(default_factory=list)       # Nested classes defined in class body
    attributes: Dict[str, Any] = field(default_factory=dict)        # Class attributes, e.g. {'hidden_size': 768, ...}
    docstring: Optional[str] = None                                 # Class docstring
    decorators: List[str] = field(default_factory=list)             # Class decorators, e.g. ['@dataclass']
    line_number: int = 0  # Class definition line (disambiguate same names)
    col_offset: int = 0  # Column offset (indentation)
    full_id: str = ""  # Full identifier: file_path::qualified_name::line_number
    content: Optional[str] = None  # Source snippet


@dataclass
class ModuleSignature:
    """Module signature information."""
    file_path: str = ""      # File path, e.g. '/testbed/src/model.py' or '/project/utils/helper.py'
    imports: List[str] = field(default_factory=list)    # import statements, e.g. ['import torch', 'from typing import Dict']
    constants: Dict[str, Any] = field(default_factory=dict) # Module-level constants, e.g. {'MAX_LENGTH': 512, ...}
    docstring: Optional[str] = None # Module docstring
    functions: List[FunctionSignature] = field(default_factory=list)    # Top-level function signatures (no nested funcs)
    classes: List[ClassSignature] = field(default_factory=list) # Top-level class signatures (no nested classes)
    content: Optional[str] = None  # Full module source


class SignatureExtractor(ast.NodeVisitor):
    """Extract function/class signatures from AST."""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        self.module_signature = ModuleSignature()
        self.scope_stack = []  # Scope stack: current scope name path
        self.scope_type_stack = []  # Scope type stack: 'class' or 'function'
        self.class_stack = []  # Class stack: current ClassSignature
        self.function_stack = []  # Function stack: current FunctionSignature
        
    def visit_Module(self, node: ast.Module):
        """Visit module node."""
        # Extract module docstring
        if (node.body and isinstance(node.body[0], ast.Expr) 
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            self.module_signature.docstring = node.body[0].value.value
        # Record full module source
        self.module_signature.content = self.source_code
            
        self.generic_visit(node)
        
    def visit_Import(self, node: ast.Import):
        """Visit import statement."""
        for alias in node.names:
            import_str = f"import {alias.name}"
            if alias.asname:
                import_str += f" as {alias.asname}"
            self.module_signature.imports.append(import_str)
            
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from ... import statement."""
        module = node.module or ""
        for alias in node.names:
            import_str = f"from {module} import {alias.name}"
            if alias.asname:
                import_str += f" as {alias.asname}"
            self.module_signature.imports.append(import_str)
            
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition."""
        func_sig = self._process_function(node, is_async=False)
        
        # Enter function scope
        self.scope_stack.append(node.name)
        self.scope_type_stack.append('function')
        self.function_stack.append(func_sig)
        
        # Recursively visit nested definitions in function body
        self.generic_visit(node)
        
        # Exit function scope
        self.scope_stack.pop()
        self.scope_type_stack.pop()
        self.function_stack.pop()
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition."""
        func_sig = self._process_function(node, is_async=True)
        
        # Enter function scope
        self.scope_stack.append(node.name)
        self.scope_type_stack.append('function')
        self.function_stack.append(func_sig)
        
        # Recursively visit nested definitions in function body
        self.generic_visit(node)
        
        # Exit function scope
        self.scope_stack.pop()
        self.scope_type_stack.pop()
        self.function_stack.pop()
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        # Build qualified name from scope_stack
        if self.scope_stack:
            qualified_name = ".".join(self.scope_stack) + "." + node.name
        else:
            qualified_name = node.name
            
        class_sig = ClassSignature(
            name=node.name,
            qualified_name=qualified_name,
            line_number=node.lineno,  # Record line number to disambiguate
            col_offset=node.col_offset  # Record column offset (indentation)
        )
        
        # Extract base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                class_sig.bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                class_sig.bases.append(ast.unparse(base))
                
        # Extract decorators
        for decorator in node.decorator_list:
            class_sig.decorators.append(ast.unparse(decorator))
            
        # Extract docstring
        if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            class_sig.docstring = node.body[0].value.value

        # Record class source snippet
        class_sig.content = self._get_node_source(node)
        
        # Extract class attributes (directly defined in class body)
        for child in node.body:
            if isinstance(child, ast.Assign):
                # Extract class attributes (plain assignments)
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        try:
                            # Try to evaluate literal
                            value = ast.literal_eval(child.value)
                            class_sig.attributes[target.id] = {
                                "_type": "literal",
                                "_value": value
                            }
                        except (ValueError, TypeError):
                            # Non-literal expression: keep source text
                            class_sig.attributes[target.id] = {
                                "_type": "expression",
                                "_code": ast.unparse(child.value)
                            }
            elif isinstance(child, ast.AnnAssign):
                # Extract class attributes (typed assignments)
                if isinstance(child.target, ast.Name):
                    attr_name = child.target.id
                    if child.value:
                        # Type annotation with value
                        try:
                            value = ast.literal_eval(child.value)
                            class_sig.attributes[attr_name] = {
                                "_type": "literal",
                                "_value": value,
                                "_annotation": ast.unparse(child.annotation)
                            }
                        except (ValueError, TypeError):
                            class_sig.attributes[attr_name] = {
                                "_type": "expression",
                                "_code": ast.unparse(child.value),
                                "_annotation": ast.unparse(child.annotation)
                            }
                    else:
                        # Type annotation only; keep annotation (for dataclass, etc.)
                        class_sig.attributes[attr_name] = {
                            "_type": "annotation_only",
                            "_annotation": ast.unparse(child.annotation)
                        }
        
        # Decide where to attach the class
        # Use scope_type_stack to detect nearest scope
        if not self.scope_type_stack:
            # Top-level class -> module classes
            self.module_signature.classes.append(class_sig)
        elif self.scope_type_stack[-1] == 'function':
            # Nearest scope is function -> parent function nested_classes
            self.function_stack[-1].nested_classes.append(class_sig)
        elif self.scope_type_stack[-1] == 'class':
            # Nearest scope is class -> parent class nested_classes
            self.class_stack[-1].nested_classes.append(class_sig)
            
        # Enter class scope
        self.scope_stack.append(node.name)
        self.scope_type_stack.append('class')
        self.class_stack.append(class_sig)
        
        # Recursively visit all definitions in class body
        self.generic_visit(node)
        
        # Exit class scope
        self.scope_stack.pop()
        self.scope_type_stack.pop()
        self.class_stack.pop()
        
    def visit_Assign(self, node: ast.Assign):
        """Visit assignment statement (module-level constants)."""
        # Only handle module-level assignments (not inside classes/functions)
        if not self.class_stack and not self.function_stack:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        value = ast.literal_eval(node.value)
                        self.module_signature.constants[target.id] = value
                    except (ValueError, TypeError):
                        self.module_signature.constants[target.id] = ast.unparse(node.value)
                        
    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool):
        """Handle function definition."""
        # Build qualified name from scope_stack
        if self.scope_stack:
            qualified_name = ".".join(self.scope_stack) + "." + node.name
        else:
            qualified_name = node.name
            
        func_sig = FunctionSignature(
            name=node.name,
            qualified_name=qualified_name,
            is_async=is_async,
            line_number=node.lineno,  # Record line number to disambiguate
            col_offset=node.col_offset  # Record column offset (indentation)
        )
        
        # Extract function parameters
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
            
        # Handle default parameters
        defaults_offset = len(args) - len(node.args.defaults)
        for i, default in enumerate(node.args.defaults):
            default_value = ast.unparse(default)
            target_index = defaults_offset + i
            if 0 <= target_index < len(args):
                args[target_index] += f" = {default_value}"
            
        # Handle *args
        if node.args.vararg:
            vararg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(vararg_str)
            
        # Handle **kwargs
        if node.args.kwarg:
            kwarg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(kwarg_str)
            
        func_sig.args = args
        
        # Extract return annotation
        if node.returns:
            func_sig.return_annotation = ast.unparse(node.returns)
            
        # Extract docstring
        if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            func_sig.docstring = node.body[0].value.value

        # Record function source snippet
        func_sig.content = self._get_node_source(node)
            
        # Extract decorators
        for decorator in node.decorator_list:
            decorator_str = ast.unparse(decorator)
            func_sig.decorators.append(decorator_str)
                
        # Decide where to attach the function
        # Use scope_type_stack to detect nearest scope
        if not self.scope_type_stack:
            # Top-level function -> module functions
            self.module_signature.functions.append(func_sig)
        elif self.scope_type_stack[-1] == 'function':
            # Nearest scope is function -> parent function nested_functions
            self.function_stack[-1].nested_functions.append(func_sig)
        elif self.scope_type_stack[-1] == 'class':
            # Nearest scope is class -> class nested_functions
            self.class_stack[-1].nested_functions.append(func_sig)
        
        return func_sig

    # ------------------------ helpers ------------------------

    def _get_node_start(self, node: ast.AST) -> int:
        """Get node start line (including decorators), 0-based."""
        if getattr(node, "decorator_list", None):
            return node.decorator_list[0].lineno - 1
        return node.lineno - 1

    def _get_node_end(self, node: ast.AST) -> int:
        """Get node end line, 0-based."""
        if hasattr(node, "end_lineno") and node.end_lineno:
            return node.end_lineno - 1
        start_line = node.lineno - 1
        if start_line >= len(self.source_lines):
            return start_line
        base_indent = len(self.source_lines[start_line]) - len(self.source_lines[start_line].lstrip())
        cur = start_line + 1
        while cur < len(self.source_lines):
            line = self.source_lines[cur]
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent <= base_indent:
                    return cur - 1
            cur += 1
        return len(self.source_lines) - 1

    def _get_node_source(self, node: ast.AST) -> str:
        """Return source segment for node (decorators + body)."""
        start = self._get_node_start(node)
        end = self._get_node_end(node)
        return "\n".join(self.source_lines[start : end + 1])


def extract_signature(source_code: str, file_path: str = "") -> ModuleSignature:
    """
    Extract signature information from source code.
    
    Args:
        source_code: Python source code
        file_path: File path
        
    Returns:
        ModuleSignature: Extracted signature information
    """
    try:
        tree = ast.parse(source_code)
        extractor = SignatureExtractor(source_code)
        extractor.visit(tree)
        extractor.module_signature.file_path = file_path
        
        # Set full_id for all functions and classes
        _set_full_ids(extractor.module_signature)
        
        return extractor.module_signature
    except Exception as e:
        # On parse failure, return empty signature info
        return ModuleSignature(file_path=file_path)


def _set_full_ids(module_sig: ModuleSignature):
    """Recursively assign full_id for all functions and classes."""
    file_path = module_sig.file_path
    
    # Handle top-level functions
    for func in module_sig.functions:
        _set_function_full_id(func, file_path)
    
    # Handle top-level classes
    for cls in module_sig.classes:
        _set_class_full_id(cls, file_path)

def _set_function_full_id(func: FunctionSignature, file_path: str):
    """Recursively assign full_id for function and nested objects."""
    func.full_id = f"{file_path}::{func.qualified_name}::{func.line_number}"
    
    # Handle nested functions
    for nested_func in func.nested_functions:
        _set_function_full_id(nested_func, file_path)
    
    # Handle nested classes
    for nested_cls in func.nested_classes:
        _set_class_full_id(nested_cls, file_path)


def _set_class_full_id(cls: ClassSignature, file_path: str):
    """Recursively assign full_id for class and nested objects."""
    cls.full_id = f"{file_path}::{cls.qualified_name}::{cls.line_number}"
    
    # Handle nested functions (methods)
    for method in cls.nested_functions:
        _set_function_full_id(method, file_path)
    
    # Handle nested classes
    for nested_cls in cls.nested_classes:
        _set_class_full_id(nested_cls, file_path)
