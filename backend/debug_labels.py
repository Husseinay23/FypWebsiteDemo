"""
Debug script to print label indices and verify label order.
Run this and compare with training notebook output.
"""
from backend.labels_22 import DIALECT_LABELS

if __name__ == "__main__":
    print("Label Index -> Label Name")
    print("=" * 40)
    for idx, name in enumerate(DIALECT_LABELS):
        print(f"{idx:2d} -> {name}")
    
    print("\n" + "=" * 40)
    print(f"Total labels: {len(DIALECT_LABELS)}")
    print("\nCompare this output with idx_to_label from training notebook.")
    print("The indices MUST match exactly!")

