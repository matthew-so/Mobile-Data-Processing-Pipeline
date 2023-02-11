import sys
import warnings # Optionally suppress warnings for debugging
sys.path.insert(0, '../../')
from src.main import main

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
