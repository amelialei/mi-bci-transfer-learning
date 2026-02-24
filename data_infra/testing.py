import numpy as np
from data_infra.config import *
from data_infra.preprocessing import get_subject_data

def test_pipeline():
    """
    Run this to check if your code works!
    """
    print("=" * 60)
    print("TESTING DATA PIPELINE")
    print("=" * 60)
    print()
    
    # Test Dataset 2a
    try:
        print("Test 1: Dataset 2a (Subject 1)")
        print("-" * 40)
        X_train, X_val, y_train, y_val = get_subject_data('2a', subject_id=1)
        
        # Check if everything looks correct
        assert X_train.shape[1] == 22, "Should have 22 channels!"
        assert len(np.unique(y_train)) == 4, "Should have 4 classes!"
        
        print("Dataset 2a works perfectly!")
        print()
        
    except Exception as e:
        print(f"Dataset 2a failed: {e}")
        print()
    
    # Test Dataset 2b
    try:
        print("Test 2: Dataset 2b (Subject 1)")
        print("-" * 40)
        X_train, X_val, y_train, y_val = get_subject_data('2b', subject_id=1)
        
        # Check if everything looks correct
        assert X_train.shape[1] == 3, "Should have 3 channels!"
        assert len(np.unique(y_train)) == 2, "Should have 2 classes!"
        
        print("✅ Dataset 2b works perfectly!")
        print()
        
    except Exception as e:
        print(f"Dataset 2b failed: {e}")
        print()
    
    print("=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60)


# This runs when you execute the file directly
if __name__ == "__main__":
    test_pipeline()
