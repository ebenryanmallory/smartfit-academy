// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson2: LessonData = {
  id: 2,
  title: "Advanced Programming Paradigms and Language Theory",
  description: "Comprehensive exploration of programming language design, computational models, and advanced Python implementation techniques for research and industry applications.",
  sections: [
    {
      title: "Computational Models and Programming Language Theory",
      content: `Programming languages serve as formal systems for expressing computational procedures, grounded in mathematical models of computation:

**Lambda Calculus and Functional Foundations**
- Church's lambda calculus as the theoretical foundation for functional programming
- Combinatory logic and the SKI combinator calculus
- Curry-Howard correspondence between types and propositions
- Denotational semantics and domain theory

**Turing Machines and Imperative Models**
- Von Neumann architecture and the stored-program concept
- State transformation as the imperative programming paradigm
- Operational semantics and small-step/big-step evaluation

**Type Theory and Category Theory**
- Simply typed lambda calculus and polymorphic type systems
- Hindley-Milner type inference algorithm
- Dependent types and proof assistants
- Categorical semantics and functorial programming

Python's design philosophy reflects pragmatic choices balancing expressiveness, readability, and performance, incorporating elements from multiple paradigms while maintaining dynamic typing with optional static analysis.`
    },
    {
      title: "Advanced Data Models and Memory Management",
      content: `Python's object model implements sophisticated abstractions over lower-level memory management:

**Object Identity and Reference Semantics**
\`\`\`python interactive
import sys
import gc
from weakref import WeakValueDictionary

class MetaclassExample(type):
    def __new__(mcs, name, bases, namespace):
        # Metaclass modifies class creation
        namespace['class_id'] = id(namespace)
        return super().__new__(mcs, name, bases, namespace)

class AdvancedContainer(metaclass=MetaclassExample):
    __slots__ = ['_data', '_metadata']  # Memory optimization
    
    def __init__(self, data):
        self._data = data
        self._metadata = {'created_at': __import__('time').time()}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Resource cleanup in context manager protocol
        if hasattr(self._data, 'close'):
            self._data.close()
        return False

# Demonstration of memory behavior
container = AdvancedContainer([1, 2, 3, 4, 5])
print(f"Object ID: {id(container)}")
print(f"Class ID: {container.class_id}")
print(f"Reference count: {sys.getrefcount(container)}")

# Weak references for avoiding circular dependencies
weak_refs = WeakValueDictionary()
weak_refs['container'] = container
print(f"Weak reference exists: {'container' in weak_refs}")
\`\`\`

**Memory Layout and Performance Optimization**
- Object header overhead and memory fragmentation
- Reference counting vs. generational garbage collection
- Copy-on-write semantics and immutable data structures
- Memory profiling and leak detection techniques`
    },
    {
      title: "Type System Design and Static Analysis",
      content: `Modern Python incorporates gradual typing through PEP 484, enabling sophisticated static analysis:

**Advanced Type Annotations and Generic Programming**
\`\`\`python interactive
from typing import (
    TypeVar, Generic, Protocol, Union, Optional, 
    Callable, Iterator, Mapping, Sequence, runtime_checkable
)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch, wraps

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other) -> bool: ...
    def __eq__(self, other) -> bool: ...

class BinarySearchTree(Generic[T]):
    """Generic binary search tree with protocol-based constraints."""
    
    @dataclass
    class Node:
        value: T
        left: Optional['BinarySearchTree.Node[T]'] = None
        right: Optional['BinarySearchTree.Node[T]'] = None
    
    def __init__(self) -> None:
        self.root: Optional[self.Node[T]] = None
    
    def insert(self, value: T) -> None:
        """Insert value maintaining BST invariant."""
        if not isinstance(value, Comparable):
            raise TypeError("Values must implement Comparable protocol")
        self.root = self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node: Optional[Node[T]], value: T) -> Node[T]:
        if node is None:
            return self.Node(value)
        
        if value < node.value:
            node.left = self._insert_recursive(node.left, value)
        elif value > node.value:
            node.right = self._insert_recursive(node.right, value)
        
        return node
    
    def inorder_traversal(self) -> Iterator[T]:
        """Yield values in sorted order."""
        yield from self._inorder_recursive(self.root)
    
    def _inorder_recursive(self, node: Optional[Node[T]]) -> Iterator[T]:
        if node is not None:
            yield from self._inorder_recursive(node.left)
            yield node.value
            yield from self._inorder_recursive(node.right)

# Type-safe usage
bst: BinarySearchTree[int] = BinarySearchTree()
for value in [5, 3, 7, 1, 9, 4, 6]:
    bst.insert(value)

print(f"Sorted values: {list(bst.inorder_traversal())}")
\`\`\`

**Dependent Types and Runtime Validation**
\`\`\`python interactive
from typing import Annotated, get_type_hints
from pydantic import BaseModel, validator, Field
import inspect

def positive(x):
    """Runtime constraint for positive numbers."""
    return x > 0

def bounded(min_val, max_val):
    """Runtime constraint factory for bounded values."""
    def constraint(x):
        return min_val <= x <= max_val
    return constraint

PositiveInt = Annotated[int, positive]
Percentage = Annotated[float, bounded(0.0, 100.0)]

class AdvancedModel(BaseModel):
    """Pydantic model with runtime type validation."""
    student_id: PositiveInt
    grade_percentage: Percentage
    courses: Sequence[str] = Field(min_items=1, max_items=10)
    
    @validator('student_id')
    def validate_student_id(cls, v):
        if v <= 0:
            raise ValueError('Student ID must be positive')
        return v
    
    @validator('grade_percentage')
    def validate_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Grade must be between 0 and 100')
        return v

# Demonstrate type-safe model creation
try:
    model = AdvancedModel(
        student_id=12345,
        grade_percentage=87.5,
        courses=["Advanced Algorithms", "Machine Learning"]
    )
    print(f"Valid model: {model}")
except Exception as e:
    print(f"Validation error: {e}")
\`\`\``
    },
    {
      title: "Functional Programming and Monadic Patterns",
      content: `Python supports functional programming paradigms through higher-order functions, closures, and monadic design patterns:

**Higher-Order Functions and Closure Composition**
\`\`\`python interactive
from functools import reduce, partial, wraps
from typing import Callable, TypeVar, Generic, Union
from abc import ABC, abstractmethod

F = TypeVar('F', bound=Callable)

def memoize(func: F) -> F:
    """Memoization decorator with closure-based cache."""
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache = cache
    wrapper.cache_clear = cache.clear
    return wrapper

def curry(func: Callable) -> Callable:
    """Transform function into curried form."""
    @wraps(func)
    def curried(*args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        
        if len(bound.arguments) >= len(sig.parameters):
            return func(*args, **kwargs)
        else:
            return lambda *more_args, **more_kwargs: curried(
                *(args + more_args), **{**kwargs, **more_kwargs}
            )
    return curried

# Function composition and pipeline operations
def compose(*functions):
    """Compose functions right-to-left: compose(f, g, h)(x) = f(g(h(x)))"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def pipe(value, *functions):
    """Apply functions left-to-right: pipe(x, f, g, h) = h(g(f(x)))"""
    return reduce(lambda acc, func: func(acc), functions, value)

# Practical example with data processing pipeline
@memoize
@curry
def multiply(a: float, b: float) -> float:
    return a * b

@curry
def add(a: float, b: float) -> float:
    return a + b

@curry
def power(base: float, exponent: float) -> float:
    return base ** exponent

# Create specialized functions through partial application
double = multiply(2)
square = power(2)
add_ten = add(10)

# Compose complex transformations
transform = compose(add_ten, square, double)
result = transform(5)  # ((5 * 2) ** 2) + 10 = 110
print(f"Composed result: {result}")

# Pipeline processing
data_pipeline = lambda x: pipe(
    x,
    double,           # x * 2
    square,          # (x * 2) ** 2
    add_ten          # ((x * 2) ** 2) + 10
)
print(f"Pipeline result: {data_pipeline(5)}")
\`\`\`

**Monadic Design Patterns**
\`\`\`python interactive
from typing import Optional, Callable, TypeVar, Generic, Union
from abc import ABC, abstractmethod

A = TypeVar('A')
B = TypeVar('B')

class Monad(Generic[A], ABC):
    """Abstract base class for monadic types."""
    
    @abstractmethod
    def bind(self, func: Callable[[A], 'Monad[B]']) -> 'Monad[B]':
        """Monadic bind operation (>>=)."""
        pass
    
    @classmethod
    @abstractmethod
    def unit(cls, value: A) -> 'Monad[A]':
        """Monadic unit operation (return)."""
        pass
    
    def map(self, func: Callable[[A], B]) -> 'Monad[B]':
        """Functor map operation."""
        return self.bind(lambda x: self.unit(func(x)))

class Maybe(Monad[A]):
    """Maybe monad for handling null values safely."""
    
    def __init__(self, value: Optional[A]):
        self._value = value
    
    def bind(self, func: Callable[[A], 'Maybe[B]']) -> 'Maybe[B]':
        if self._value is None:
            return Maybe(None)
        return func(self._value)
    
    @classmethod
    def unit(cls, value: A) -> 'Maybe[A]':
        return cls(value)
    
    @property
    def value(self) -> Optional[A]:
        return self._value
    
    def is_nothing(self) -> bool:
        return self._value is None

# Demonstration of monadic composition
def safe_divide(x: float, y: float) -> Maybe[float]:
    if y == 0:
        return Maybe(None)
    return Maybe(x / y)

def safe_sqrt(x: float) -> Maybe[float]:
    if x < 0:
        return Maybe(None)
    return Maybe(x ** 0.5)

def safe_log(x: float) -> Maybe[float]:
    if x <= 0:
        return Maybe(None)
    return Maybe(__import__('math').log(x))

# Chain operations safely without explicit null checking
result = (Maybe(16.0)
    .bind(lambda x: safe_divide(x, 4))  # 16 / 4 = 4
    .bind(safe_sqrt)                    # sqrt(4) = 2
    .bind(safe_log))                    # log(2) ≈ 0.693

print(f"Monadic result: {result.value}")

# Demonstrate short-circuiting on None
invalid_result = (Maybe(16.0)
    .bind(lambda x: safe_divide(x, 0))  # Division by zero
    .bind(safe_sqrt)                    # Never executed
    .bind(safe_log))                    # Never executed

print(f"Invalid result: {invalid_result.value}")
\`\`\``
    },
    {
      title: "Concurrent Programming and Asynchronous Patterns",
      content: `Modern Python provides sophisticated concurrency primitives through asyncio, threading, and multiprocessing:

**Asynchronous Programming with Coroutines**
\`\`\`python interactive
import asyncio
import aiohttp
import time
from typing import List, Awaitable, Callable, TypeVar
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps

T = TypeVar('T')

def async_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for async functions with exponential backoff retry."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        await asyncio.sleep(wait_time)
            
            raise last_exception
        return wrapper
    return decorator

class AsyncResourcePool:
    """Asynchronous resource pool with semaphore-based concurrency control."""
    
    def __init__(self, max_size: int = 10):
        self._semaphore = asyncio.Semaphore(max_size)
        self._resources = asyncio.Queue(maxsize=max_size)
        self._created_count = 0
    
    async def acquire(self) -> int:
        await self._semaphore.acquire()
        try:
            resource = self._resources.get_nowait()
        except asyncio.QueueEmpty:
            # Create new resource if pool is empty
            resource = self._created_count
            self._created_count += 1
        return resource
    
    async def release(self, resource: int):
        await self._resources.put(resource)
        self._semaphore.release()
    
    async def __aenter__(self):
        self._resource = await self.acquire()
        return self._resource
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release(self._resource)

@async_retry(max_attempts=3, delay=0.5)
async def fetch_data(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch data from URL with retry logic."""
    async with session.get(url) as response:
        if response.status != 200:
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=response.status
            )
        return await response.json()

async def parallel_data_processing():
    """Demonstrate parallel async processing with resource pooling."""
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2", 
        "https://httpbin.org/json"
    ] * 3  # Simulate multiple requests
    
    pool = AsyncResourcePool(max_size=5)
    
    async with aiohttp.ClientSession() as session:
        async def process_url(url: str) -> dict:
            async with pool:  # Automatic resource management
                print(f"Processing {url} with resource {pool._resource}")
                return await fetch_data(session, url)
        
        # Execute requests concurrently with controlled parallelism
        start_time = time.time()
        results = await asyncio.gather(
            *[process_url(url) for url in urls],
            return_exceptions=True
        )
        end_time = time.time()
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        print(f"Processed {len(urls)} requests in {end_time - start_time:.2f}s")
        print(f"Successful: {len(successful_results)}, Failed: {len(failed_results)}")

# Run the async example (note: would need to be called with asyncio.run() in practice)
print("Async concurrency example prepared for execution")
\`\`\`

**Advanced Threading and Multiprocessing Patterns**
\`\`\`python interactive
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, PriorityQueue
from dataclasses import dataclass, field
from typing import Callable, Any, Iterator
import time

@dataclass
class Task:
    priority: int
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.priority < other.priority

class AdvancedThreadPool:
    """Thread pool with priority queue and dynamic scaling."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 10):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.task_queue = PriorityQueue()
        self.workers = []
        self.shutdown_event = threading.Event()
        self.active_tasks = threading.Semaphore(0)
        
        # Start minimum number of worker threads
        for _ in range(min_workers):
            self._start_worker()
    
    def _start_worker(self):
        """Start a new worker thread."""
        def worker():
            while not self.shutdown_event.is_set():
                try:
                    task = self.task_queue.get(timeout=1.0)
                    try:
                        result = task.func(*task.args, **task.kwargs)
                        print(f"Task completed by {threading.current_thread().name}")
                    except Exception as e:
                        print(f"Task failed: {e}")
                    finally:
                        self.task_queue.task_done()
                        self.active_tasks.release()
                except:
                    continue  # Timeout or shutdown
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.workers.append(thread)
    
    def submit(self, func: Callable, *args, priority: int = 5, **kwargs):
        """Submit task with priority."""
        task = Task(priority, func, args, kwargs)
        self.task_queue.put(task)
        self.active_tasks.acquire(blocking=False)
        
        # Dynamic scaling: add worker if queue is growing
        if (self.task_queue.qsize() > len(self.workers) and 
            len(self.workers) < self.max_workers):
            self._start_worker()
    
    def shutdown(self):
        """Gracefully shutdown the thread pool."""
        self.task_queue.join()  # Wait for all tasks to complete
        self.shutdown_event.set()
        for worker in self.workers:
            worker.join(timeout=1.0)

# Example computational tasks for parallel processing
def cpu_intensive_task(n: int) -> int:
    """Simulate CPU-intensive computation."""
    total = 0
    for i in range(n * 1000000):
        total += i ** 0.5
    return int(total)

def io_intensive_task(duration: float) -> str:
    """Simulate I/O-intensive task."""
    time.sleep(duration)
    return f"IO task completed after {duration}s"

# Demonstrate optimal task distribution
def demonstrate_concurrency():
    """Compare threading vs multiprocessing for different task types."""
    
    # CPU-intensive tasks: better with multiprocessing
    cpu_tasks = [100, 200, 150, 300, 250]
    
    print("CPU-intensive tasks with ProcessPoolExecutor:")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        cpu_futures = [executor.submit(cpu_intensive_task, n) for n in cpu_tasks]
        cpu_results = [f.result() for f in as_completed(cpu_futures)]
    cpu_time = time.time() - start_time
    print(f"Multiprocessing time: {cpu_time:.2f}s")
    
    # I/O-intensive tasks: better with threading
    io_tasks = [0.5, 1.0, 0.7, 1.2, 0.8]
    
    print("\\nI/O-intensive tasks with ThreadPoolExecutor:")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        io_futures = [executor.submit(io_intensive_task, d) for d in io_tasks]
        io_results = [f.result() for f in as_completed(io_futures)]
    io_time = time.time() - start_time
    print(f"Threading time: {io_time:.2f}s")

print("Concurrency demonstration prepared for execution")
\`\`\``
    },
    {
      title: "Metaprogramming and Code Generation",
      content: `Python's introspection capabilities enable sophisticated metaprogramming techniques for code generation and runtime manipulation:

**Advanced Metaclass Programming**
\`\`\`python interactive
import inspect
import ast
import types
from typing import Dict, Any, Callable, Type
from functools import wraps

class SingletonMeta(type):
    """Metaclass implementing thread-safe singleton pattern."""
    _instances = {}
    _lock = __import__('threading').Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ValidatedAttributeMeta(type):
    """Metaclass that adds runtime validation to class attributes."""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Extract validation rules from annotations
        annotations = namespace.get('__annotations__', {})
        validators = {}
        
        for attr_name, attr_type in annotations.items():
            if hasattr(attr_type, '__origin__') and hasattr(attr_type, '__args__'):
                # Handle generic types with constraints
                validators[attr_name] = mcs._create_validator(attr_type)
        
        # Create property descriptors for validated attributes
        for attr_name, validator in validators.items():
            private_name = f'_{attr_name}'
            
            def make_property(attr, priv_name, val_func):
                def getter(self):
                    return getattr(self, priv_name, None)
                
                def setter(self, value):
                    if val_func and not val_func(value):
                        raise ValueError(f"Invalid value for {attr}: {value}")
                    setattr(self, priv_name, value)
                
                return property(getter, setter)
            
            namespace[attr_name] = make_property(attr_name, private_name, validator)
        
        return super().__new__(mcs, name, bases, namespace)
    
    @staticmethod
    def _create_validator(type_hint) -> Callable[[Any], bool]:
        """Create validation function from type hint."""
        def validator(value):
            try:
                # Basic type checking - can be extended for complex constraints
                return isinstance(value, type_hint) if not hasattr(type_hint, '__origin__') else True
            except:
                return False
        return validator

class DatabaseORM(metaclass=ValidatedAttributeMeta):
    """Example ORM-like class with automatic validation."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def create_table_sql(cls) -> str:
        """Generate SQL CREATE TABLE statement from class definition."""
        annotations = getattr(cls, '__annotations__', {})
        type_mapping = {
            int: 'INTEGER',
            str: 'VARCHAR(255)',
            float: 'REAL',
            bool: 'BOOLEAN'
        }
        
        columns = []
        for attr_name, attr_type in annotations.items():
            sql_type = type_mapping.get(attr_type, 'TEXT')
            columns.append(f"{attr_name} {sql_type}")
        
        return f"CREATE TABLE {cls.__name__.lower()} ({', '.join(columns)});"

class User(DatabaseORM):
    """Example model class with automatic SQL generation."""
    user_id: int
    username: str
    email: str
    is_active: bool

print(f"Generated SQL: {User.create_table_sql()}")

# Demonstrate validation
try:
    user = User(user_id=123, username="john_doe", email="john@example.com", is_active=True)
    print(f"Valid user created: {user.username}")
except ValueError as e:
    print(f"Validation error: {e}")
\`\`\`

**Dynamic Code Generation and AST Manipulation**
\`\`\`python interactive
import ast
import inspect
import textwrap
from typing import List, Dict, Any

class CodeGenerator:
    """Advanced code generation using AST manipulation."""
    
    def __init__(self):
        self.generated_functions = {}
    
    def create_optimized_accessor(self, field_names: List[str]) -> Callable:
        """Generate optimized accessor function for specific fields."""
        
        # Build AST for optimized function
        args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='obj', annotation=None)],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        )
        
        # Generate return dictionary with field access
        keys = [ast.Constant(value=name) for name in field_names]
        values = [
            ast.Attribute(
                value=ast.Name(id='obj', ctx=ast.Load()),
                attr=name,
                ctx=ast.Load()
            ) for name in field_names
        ]
        
        return_dict = ast.Dict(keys=keys, values=values)
        
        func_def = ast.FunctionDef(
            name='optimized_accessor',
            args=args,
            body=[ast.Return(value=return_dict)],
            decorator_list=[],
            returns=None
        )
        
        # Compile and execute the generated AST
        module = ast.Module(body=[func_def], type_ignores=[])
        ast.fix_missing_locations(module)
        
        code = compile(module, '<generated>', 'exec')
        namespace = {}
        exec(code, namespace)
        
        return namespace['optimized_accessor']
    
    def create_batch_processor(self, operation: str) -> Callable:
        """Generate batch processing function for specific operations."""
        
        operation_map = {
            'sum': ast.Call(
                func=ast.Name(id='sum', ctx=ast.Load()),
                args=[ast.Name(id='items', ctx=ast.Load())],
                keywords=[]
            ),
            'max': ast.Call(
                func=ast.Name(id='max', ctx=ast.Load()),
                args=[ast.Name(id='items', ctx=ast.Load())],
                keywords=[]
            ),
            'avg': ast.BinOp(
                left=ast.Call(
                    func=ast.Name(id='sum', ctx=ast.Load()),
                    args=[ast.Name(id='items', ctx=ast.Load())],
                    keywords=[]
                ),
                op=ast.Div(),
                right=ast.Call(
                    func=ast.Name(id='len', ctx=ast.Load()),
                    args=[ast.Name(id='items', ctx=ast.Load())],
                    keywords=[]
                )
            )
        }
        
        if operation not in operation_map:
            raise ValueError(f"Unsupported operation: {operation}")
        
        # Generate function with error handling
        func_body = [
            ast.If(
                test=ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Name(id='items', ctx=ast.Load())
                ),
                body=[ast.Return(value=ast.Constant(value=0))],
                orelse=[]
            ),
            ast.Return(value=operation_map[operation])
        ]
        
        args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='items', annotation=None)],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        )
        
        func_def = ast.FunctionDef(
            name=f'batch_{operation}',
            args=args,
            body=func_body,
            decorator_list=[],
            returns=None
        )
        
        module = ast.Module(body=[func_def], type_ignores=[])
        ast.fix_missing_locations(module)
        
        code = compile(module, '<generated>', 'exec')
        namespace = {}
        exec(code, namespace)
        
        return namespace[f'batch_{operation}']

# Demonstrate dynamic code generation
generator = CodeGenerator()

# Generate optimized accessor for specific fields
class DataPoint:
    def __init__(self, x, y, z, timestamp):
        self.x, self.y, self.z = x, y, z
        self.timestamp = timestamp

accessor = generator.create_optimized_accessor(['x', 'y', 'z'])
data_point = DataPoint(1.0, 2.0, 3.0, "2023-01-01")
coords = accessor(data_point)
print(f"Extracted coordinates: {coords}")

# Generate batch processors
sum_processor = generator.create_batch_processor('sum')
avg_processor = generator.create_batch_processor('avg')

test_data = [1, 2, 3, 4, 5]
print(f"Batch sum: {sum_processor(test_data)}")
print(f"Batch average: {avg_processor(test_data)}")
\`\`\``
    },
    {
      title: "Performance Optimization and Profiling",
      content: `Advanced performance optimization requires understanding Python's execution model and utilizing appropriate profiling tools:

**Algorithmic Complexity Analysis and Optimization**
\`\`\`python interactive
import cProfile
import pstats
import timeit
import memory_profiler
from functools import lru_cache
from typing import Iterator, List, Tuple
import sys

class PerformanceAnalyzer:
    """Comprehensive performance analysis toolkit."""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.benchmarks = {}
    
    def profile_function(self, func, *args, **kwargs):
        """Profile function execution with detailed statistics."""
        self.profiler.enable()
        result = func(*args, **kwargs)
        self.profiler.disable()
        
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        return result, stats
    
    def benchmark_alternatives(self, implementations: Dict[str, Callable], 
                             test_input, number: int = 1000):
        """Benchmark multiple implementations of the same algorithm."""
        results = {}
        
        for name, impl in implementations.items():
            time_taken = timeit.timeit(
                lambda: impl(test_input),
                number=number
            )
            results[name] = time_taken / number  # Average time per call
        
        return sorted(results.items(), key=lambda x: x[1])

# Example: Fibonacci implementations with different complexities
def fibonacci_recursive(n: int) -> int:
    """O(2^n) - Exponential time complexity."""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

@lru_cache(maxsize=None)
def fibonacci_memoized(n: int) -> int:
    """O(n) - Linear time with memoization."""
    if n <= 1:
        return n
    return fibonacci_memoized(n - 1) + fibonacci_memoized(n - 2)

def fibonacci_iterative(n: int) -> int:
    """O(n) - Linear time, O(1) space."""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fibonacci_matrix(n: int) -> int:
    """O(log n) - Logarithmic time using matrix exponentiation."""
    if n <= 1:
        return n
    
    def matrix_mult(A, B):
        return [[A[0][0]*B[0][0] + A[0][1]*B[1][0],
                 A[0][0]*B[0][1] + A[0][1]*B[1][1]],
                [A[1][0]*B[0][0] + A[1][1]*B[1][0],
                 A[1][0]*B[0][1] + A[1][1]*B[1][1]]]
    
    def matrix_power(mat, power):
        if power == 1:
            return mat
        if power % 2 == 0:
            half_pow = matrix_power(mat, power // 2)
            return matrix_mult(half_pow, half_pow)
        return matrix_mult(mat, matrix_power(mat, power - 1))
    
    base_matrix = [[1, 1], [1, 0]]
    result_matrix = matrix_power(base_matrix, n)
    return result_matrix[0][1]

# Performance comparison
analyzer = PerformanceAnalyzer()

# Test with small input for all algorithms
small_n = 20
implementations = {
    'recursive': fibonacci_recursive,
    'memoized': fibonacci_memoized,
    'iterative': fibonacci_iterative,
    'matrix': fibonacci_matrix
}

print(f"Fibonacci({small_n}) performance comparison:")
benchmark_results = analyzer.benchmark_alternatives(implementations, small_n, number=100)

for name, avg_time in benchmark_results:
    print(f"{name:12}: {avg_time*1000:.4f} ms")

# Memory profiling for data structure optimization
@memory_profiler.profile
def memory_intensive_operation():
    """Demonstrate memory usage patterns."""
    # Inefficient: list of lists
    matrix_list = [[0 for _ in range(1000)] for _ in range(1000)]
    
    # More efficient: flat list with indexing
    matrix_flat = [0] * (1000 * 1000)
    
    # Most efficient: generator for sparse data
    def sparse_matrix_generator():
        for i in range(1000):
            for j in range(1000):
                if (i + j) % 10 == 0:  # Only 10% of elements are non-zero
                    yield (i, j, i + j)
    
    sparse_data = list(sparse_matrix_generator())
    return len(sparse_data)

print(f"\\nMemory-efficient sparse matrix size: {memory_intensive_operation()}")
\`\`\`

**Advanced Caching and Memoization Strategies**
\`\`\`python interactive
import weakref
import threading
import time
from typing import Dict, Any, Optional, Callable, TypeVar
from functools import wraps
from collections import OrderedDict

T = TypeVar('T')

class AdvancedCache:
    """Thread-safe LRU cache with TTL and weak references."""
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check TTL expiration
            if self.ttl and time.time() - self._timestamps[key] > self.ttl:
                self._evict(key)
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._hits += 1
            return value
    
    def put(self, key: Any, value: Any):
        with self._lock:
            if key in self._cache:
                # Update existing key
                self._cache.pop(key)
            elif len(self._cache) >= self.maxsize:
                # Evict least recently used
                oldest_key = next(iter(self._cache))
                self._evict(oldest_key)
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def _evict(self, key: Any):
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def stats(self) -> Dict[str, int]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'size': len(self._cache)
        }

def advanced_memoize(maxsize: int = 128, ttl: Optional[float] = None, 
                    typed: bool = False):
    """Advanced memoization decorator with TTL and type-awareness."""
    def decorator(func: Callable) -> Callable:
        cache = AdvancedCache(maxsize, ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = (args, tuple(sorted(kwargs.items())))
            if typed:
                key = (key, tuple(type(arg) for arg in args))
            
            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        wrapper.cache = cache
        wrapper.cache_info = cache.stats
        wrapper.cache_clear = lambda: setattr(cache, '_cache', OrderedDict())
        return wrapper
    return decorator

# Demonstrate advanced caching
@advanced_memoize(maxsize=64, ttl=5.0, typed=True)
def expensive_computation(n: int, precision: float = 1e-6) -> float:
    """Simulate expensive computation with configurable precision."""
    time.sleep(0.1)  # Simulate computation time
    return sum(1.0 / (i ** 2) for i in range(1, n + 1))

# Test caching behavior
print("Testing advanced memoization:")
start = time.time()
result1 = expensive_computation(1000)  # Cache miss
result2 = expensive_computation(1000)  # Cache hit
end = time.time()

print(f"First call result: {result1:.6f}")
print(f"Second call result: {result2:.6f}")
print(f"Total time: {end - start:.3f}s")
print(f"Cache stats: {expensive_computation.cache_info()}")

# Test TTL expiration
time.sleep(6)  # Wait for TTL expiration
result3 = expensive_computation(1000)  # Cache miss due to TTL
print(f"After TTL expiration: {expensive_computation.cache_info()}")
\`\`\``
    }
  ]
}; 