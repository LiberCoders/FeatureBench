import ast
from typing import List, Dict, Set, Optional, Any

class FunctionClassVisitor(ast.NodeVisitor):
    """AST visitor for extracting function/class definitions."""
    
    def __init__(self):
        self.definitions = {}  # qualified_name::start_line -> (start_line, end_line, type)
        self.scope_stack = []  # Track nested scopes (functions and classes)
    
    def visit_FunctionDef(self, node):
        """Visit a function definition."""
        start_line = node.lineno
        end_line = self._get_end_line(node)
        # Build qualified name
        qualified_name = self._get_qualified_name(node.name)
        key = f"{qualified_name}::{start_line}"
        self.definitions[key] = (start_line, end_line, "Func")
        
        # Enter function scope
        self.scope_stack.append(node.name)
        # Recursively visit nested defs in the function body
        self.generic_visit(node)
        # Exit function scope
        self.scope_stack.pop()
    
    def visit_AsyncFunctionDef(self, node):
        """Visit an async function definition."""
        start_line = node.lineno
        end_line = self._get_end_line(node)
        # Build qualified name
        qualified_name = self._get_qualified_name(node.name)
        key = f"{qualified_name}::{start_line}"
        self.definitions[key] = (start_line, end_line, "Func")
        
        # Enter function scope
        self.scope_stack.append(node.name)
        # Recursively visit nested defs in the function body
        self.generic_visit(node)
        # Exit function scope
        self.scope_stack.pop()
    
    def visit_ClassDef(self, node):
        """Visit a class definition."""
        start_line = node.lineno
        end_line = self._get_end_line(node)
        # Build qualified name
        qualified_name = self._get_qualified_name(node.name)
        key = f"{qualified_name}::{start_line}"
        self.definitions[key] = (start_line, end_line, "Class")
        
        # Enter class scope
        self.scope_stack.append(node.name)
        # Recursively visit nested defs in the class body
        self.generic_visit(node)
        # Exit class scope
        self.scope_stack.pop()
    
    def _get_qualified_name(self, name: str) -> str:
        """Build qualified name from current scope stack."""
        if self.scope_stack:
            return ".".join(self.scope_stack) + "." + name
        return name
    
    def _get_end_line(self, node):
        """Get end line number for a node."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
        
        # If end_lineno is missing, estimate one
        if hasattr(node, 'body') and node.body:
            last_stmt = node.body[-1]
            if hasattr(last_stmt, 'end_lineno') and last_stmt.end_lineno:
                return last_stmt.end_lineno
            elif hasattr(last_stmt, 'lineno'):
                return last_stmt.lineno
        
        return node.lineno

def analyze_dependencies(obj_name_with_line: str, file_path: str) -> List[str]:
    """
    Use AST analysis to find same-file dependencies for a target object.
    
    Args:
        obj_name_with_line: Fully-qualified object name with line, format "qualified_name::line"
                           e.g. "forward::125" (top-level) or "ClassName.forward::125"
        file_path: Absolute file path
        
    Returns:
        List[str]: Dependent object names, format "qualified_name::line"
    """
    dependencies = set()
    
    try:
        # Parse input qualified name and line number
        if "::" not in obj_name_with_line:
            raise ValueError(f"Invalid object name format; expected 'qualified_name::line', got: {obj_name_with_line}")
        
        qualified_name, line_str = obj_name_with_line.split("::", 1)
        target_line = int(line_str)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        # Collect all functions/classes (including nested) with qualified name and line
        defined_objects = {}  # {qualified_name: line}
        
        def collect_defined_objects(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    line_no = node.lineno
                    # Build qualified name
                    qualified = f"{prefix}.{node.name}" if prefix else node.name
                    defined_objects[qualified] = line_no
                    
                    # Recursively collect nested objects
                    if hasattr(node, 'body'):
                        collect_defined_objects(node.body, qualified)
        
        collect_defined_objects(tree.body)
        
        # Verify target object exists
        if qualified_name not in defined_objects:
            print(f"WARNING: Object '{qualified_name}' not found in file '{file_path}'; classification continues but no recursive dependency analysis for this object")
            return []
        
        # Find target AST node by qualified name and line number
        target_node = None
        
        def find_target_node(nodes, target_qualified_name, target_lineno, prefix=""):
            """Find AST node by qualified name and line number."""
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    current_qualified = f"{prefix}.{node.name}" if prefix else node.name
                    if current_qualified == target_qualified_name and node.lineno == target_lineno:
                        return node
                    # Recursively search nested objects
                    if hasattr(node, 'body'):
                        child_node = find_target_node(node.body, target_qualified_name, target_lineno, current_qualified)
                        if child_node:
                            return child_node
            return None
        
        target_node = find_target_node(tree.body, qualified_name, target_line)
        
        if target_node is None:
            print(f"WARNING: Object '{qualified_name}' not found at line {target_line} in file '{file_path}'; classification continues but no recursive dependency analysis for this object")
            return []
        
        # Get parent class name if this is a class method
        # Example: "ClassName.forward" -> "ClassName"
        parent_class_name = ".".join(qualified_name.split(".")[:-1]) if "." in qualified_name else None
        
        # Analyze calls inside the target object
        class DependencyVisitor(ast.NodeVisitor):
            def __init__(self, defined_objs: dict, parent_class: str = None):
                self.defined_objs = defined_objs  # {qualified_name: line}
                self.parent_class = parent_class  # Parent class qualified name (if any)
                self.deps = set()
            
            def visit_Call(self, node):
                # Handle function calls
                if isinstance(node.func, ast.Name):
                    # Direct function call: func()
                    func_name = node.func.id
                    if func_name in self.defined_objs:
                        line_no = self.defined_objs[func_name]
                        self.deps.add(f"{func_name}::{line_no}")
                elif isinstance(node.func, ast.Attribute):
                    # Method call: obj.method() or self.method()
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id == 'self' and self.parent_class:
                            # self.method() - class method call; use qualified name
                            method_name = node.func.attr
                            qualified_method = f"{self.parent_class}.{method_name}"
                            if qualified_method in self.defined_objs:
                                line_no = self.defined_objs[qualified_method]
                                self.deps.add(f"{qualified_method}::{line_no}")
                        else:
                            # obj.method() - object method call
                            obj_name = node.func.value.id
                            if obj_name in self.defined_objs:
                                line_no = self.defined_objs[obj_name]
                                self.deps.add(f"{obj_name}::{line_no}")
                
                self.generic_visit(node)
            
            def visit_Name(self, node):
                # Handle name references (variables, class names, etc.)
                if isinstance(node.ctx, (ast.Load, ast.Store)):
                    name = node.id
                    if name in self.defined_objs:
                        # Exclude the current object itself
                        line_no = self.defined_objs[name]
                        if line_no != target_line:
                            self.deps.add(f"{name}::{line_no}")
                
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                # Handle attribute access: obj.attr
                if isinstance(node.value, ast.Name):
                    obj_name = node.value.id
                    if obj_name in self.defined_objs:
                        line_no = self.defined_objs[obj_name]
                        self.deps.add(f"{obj_name}::{line_no}")
                
                self.generic_visit(node)
        
        # Visit target node and collect dependencies
        visitor = DependencyVisitor(defined_objects, parent_class_name)
        visitor.visit(target_node)
        dependencies.update(visitor.deps)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error when parsing file {file_path}: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid argument: {e}")
    except Exception as e:
        raise Exception(f"Error analyzing dependencies: {e}")
    
    return list(dependencies)


def build_class_info(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Build class metadata in a file, including bases and abstract methods.

    Args:
        file_path: File path

    Returns:
        Dict[str, Dict[str, Any]]: Mapping from class name to metadata
            {
                "ClassName": {
                    "bases": [base class names],
                    "defined_methods": {methods defined in the class},
                    "abstract_methods": {methods marked abstract in the class}
                },
            }
    """
    class_info: Dict[str, Dict[str, Any]] = {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
    except Exception:
        return class_info

    class _Collector(ast.NodeVisitor):
        def __init__(self):
            self.classes = {}

        def visit_ClassDef(self, node: ast.ClassDef):
            bases = []
            for base in node.bases:
                name = self._extract_base_name(base)
                if name:
                    bases.append(name)

            defined_methods = set()
            abstract_methods = set()

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined_methods.add(item.name)
                    if self._is_abstract_method(item):
                        abstract_methods.add(item.name)

            self.classes[node.name] = {
                "bases": bases,
                "defined_methods": defined_methods,
                "abstract_methods": abstract_methods,
            }

            self.generic_visit(node)

        def _extract_base_name(self, base: ast.expr) -> Optional[str]:
            if isinstance(base, ast.Name):
                return base.id
            if isinstance(base, ast.Attribute):
                return base.attr
            return None

        def _is_abstract_method(self, node: ast.AST) -> bool:
            for decorator in getattr(node, "decorator_list", []):
                if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                    return True
                if isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
                    return True
            return False

    collector = _Collector()
    collector.visit(tree)

    return collector.classes