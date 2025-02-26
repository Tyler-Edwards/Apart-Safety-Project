"""
Test registry system for mapping tests to model types and managing test metadata.
"""

from typing import Dict, List, Any, Set, Type, Optional, Callable
import inspect

# Define model type categories
MODEL_TYPES = {
    "text": ["text-classification", "token-classification", "text-generation", 
             "summarization", "translation", "question-answering", "fill-mask"],
    "vision": ["image-classification", "object-detection", "image-segmentation",
               "image-to-text", "text-to-image"],
    "audio": ["audio-classification", "text-to-speech", "speech-to-text"],
    "multimodal": ["visual-question-answering", "document-question-answering", 
                  "image-text-to-text"],
    "tabular": ["tabular-classification", "tabular-regression"],
    "all": []  # Special category for tests applicable to all model types
}

# Inverse mapping for looking up categories
MODEL_TYPE_TO_CATEGORY = {}
for category, types in MODEL_TYPES.items():
    for model_type in types:
        MODEL_TYPE_TO_CATEGORY[model_type] = category

class TestRegistry:
    """
    Registry system for managing compliance tests and their metadata.
    """
    
    def __init__(self):
        self.tests = {}  # id -> test instance
        self.test_metadata = {}  # id -> metadata dict
        self.category_index = {}  # category -> list of test ids
        self.model_type_index = {}  # model_type -> list of test ids
    
    def register_test(self, test_instance, 
                     applicable_model_types: List[str] = None,
                     is_core: bool = False,
                     priority: int = 1,
                     dependencies: List[str] = None,
                     metadata: Dict[str, Any] = None):
        """
        Register a test with the registry.
        
        Args:
            test_instance: The test instance to register
            applicable_model_types: List of model types this test applies to
            is_core: Whether this is a core test that should always run
            priority: Test priority (1-5, with 1 being highest)
            dependencies: List of test IDs this test depends on
            metadata: Additional test metadata
        """
        test_id = test_instance.test_id
        
        # Default to all model types if none specified
        if applicable_model_types is None:
            applicable_model_types = ["all"]
        
        # Store the test
        self.tests[test_id] = test_instance
        
        # Create metadata
        test_metadata = {
            "name": test_instance.name,
            "description": test_instance.description,
            "category": test_instance.category,
            "applicable_model_types": applicable_model_types,
            "is_core": is_core,
            "priority": priority,
            "dependencies": dependencies or [],
            "class_name": test_instance.__class__.__name__
        }
        
        # Add any additional metadata
        if metadata:
            test_metadata.update(metadata)
        
        self.test_metadata[test_id] = test_metadata
        
        # Update indexes
        category = test_instance.category
        if category not in self.category_index:
            self.category_index[category] = []
        self.category_index[category].append(test_id)
        
        # Add to model type index
        for model_type in applicable_model_types:
            if model_type not in self.model_type_index:
                self.model_type_index[model_type] = []
            self.model_type_index[model_type].append(test_id)
    
    def get_applicable_tests(self, model_type: str) -> List[str]:
        """
        Get test IDs applicable to the given model type.
        
        Args:
            model_type: The model type to get tests for
            
        Returns:
            List[str]: List of applicable test IDs
        """
        applicable_tests = set()
        
        # Add tests specifically for this model type
        if model_type in self.model_type_index:
            applicable_tests.update(self.model_type_index[model_type])
        
        # Add tests for the category this model type belongs to
        category = MODEL_TYPE_TO_CATEGORY.get(model_type)
        if category and category in self.model_type_index:
            applicable_tests.update(self.model_type_index[category])
        
        # Always add tests marked as applicable to all model types
        if "all" in self.model_type_index:
            applicable_tests.update(self.model_type_index["all"])
        
        # Always add core tests
        core_tests = [test_id for test_id, meta in self.test_metadata.items() 
                     if meta.get("is_core", False)]
        applicable_tests.update(core_tests)
        
        return list(applicable_tests)
    
    def get_tests_by_category(self, category: str) -> List[str]:
        """
        Get test IDs for the given category.
        
        Args:
            category: The category to get tests for
            
        Returns:
            List[str]: List of test IDs in the category
        """
        return self.category_index.get(category, [])
    
    def get_test(self, test_id: str):
        """
        Get a test instance by ID.
        
        Args:
            test_id: Test ID
            
        Returns:
            Test instance or None if not found
        """
        return self.tests.get(test_id)
    
    def get_test_metadata(self, test_id: str) -> Dict[str, Any]:
        """
        Get metadata for a test.
        
        Args:
            test_id: Test ID
            
        Returns:
            Dict: Test metadata
        """
        return self.test_metadata.get(test_id, {})
    
    def get_all_test_ids(self) -> List[str]:
        """
        Get all registered test IDs.
        
        Returns:
            List[str]: List of all test IDs
        """
        return list(self.tests.keys())
    
    def get_test_dependencies(self, test_id: str) -> List[str]:
        """
        Get dependencies for a test.
        
        Args:
            test_id: Test ID
            
        Returns:
            List[str]: List of dependent test IDs
        """
        return self.test_metadata.get(test_id, {}).get("dependencies", [])
    
    def filter_tests(self, 
                    model_type: Optional[str] = None,
                    category: Optional[str] = None,
                    is_core: Optional[bool] = None,
                    min_priority: Optional[int] = None,
                    max_priority: Optional[int] = None) -> List[str]:
        """
        Filter tests based on criteria.
        
        Args:
            model_type: Filter by model type
            category: Filter by category
            is_core: Filter by core test status
            min_priority: Filter by minimum priority
            max_priority: Filter by maximum priority
            
        Returns:
            List[str]: Filtered test IDs
        """
        filtered_tests = set(self.get_all_test_ids())
        
        # Apply filters
        if model_type:
            model_type_tests = set(self.get_applicable_tests(model_type))
            filtered_tests = filtered_tests.intersection(model_type_tests)
        
        if category:
            category_tests = set(self.get_tests_by_category(category))
            filtered_tests = filtered_tests.intersection(category_tests)
        
        # Filter by metadata
        result = []
        for test_id in filtered_tests:
            metadata = self.get_test_metadata(test_id)
            
            # Check core status
            if is_core is not None and metadata.get("is_core", False) != is_core:
                continue
            
            # Check priority
            priority = metadata.get("priority", 1)
            if min_priority is not None and priority < min_priority:
                continue
            if max_priority is not None and priority > max_priority:
                continue
            
            result.append(test_id)
        
        return result

# Create a global registry instance
registry = TestRegistry()

def register(applicable_model_types=None, is_core=False, priority=1, dependencies=None, **metadata):
    """
    Decorator for registering test classes with the registry.
    
    Example:
        @register(applicable_model_types=["text", "multimodal"], is_core=True)
        class MyTest(BaseTechnicalSafetyTest):
            ...
    
    Args:
        applicable_model_types: List of applicable model types
        is_core: Whether this is a core test
        priority: Test priority (1-5)
        dependencies: List of test dependencies
        **metadata: Additional metadata
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Create an instance of the class if it's a class
        if inspect.isclass(cls):
            # We need to instantiate with test_id, name, and description
            # This assumes these are the first 3 parameters of the constructor
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.values())[1:4]  # Skip 'self'
            
            # Extract parameter names
            param_names = [p.name for p in params]
            
            # Generate default values if needed
            test_id = f"{cls.__name__.lower()}_test"
            name = cls.__name__
            description = cls.__doc__ or f"{name} Test"
            
            # Create an instance
            instance = cls(test_id, name, description)
        else:
            # Already an instance
            instance = cls
        
        # Register with the global registry
        registry.register_test(
            instance,
            applicable_model_types=applicable_model_types,
            is_core=is_core,
            priority=priority,
            dependencies=dependencies,
            metadata=metadata
        )
        
        return cls
    
    return decorator