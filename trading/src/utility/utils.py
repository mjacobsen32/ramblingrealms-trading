def read_key(path: str) -> str:
    """
    Read a key from a file.
    
    Args:
        path (str): The path to the file containing the key.
        
    Returns:
        str: The key read from the file.
    """
    try:
        with open(path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Key file not found at {path}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the key: {e}")