"""
Calculator Tool Implementation
-----------------------------
This module contains the actual execution logic for the calculator tool.
It receives the arguments provided by the LLM and computes the result.
"""

from typing import Any, Dict, Union


def execute_calculator(operation: str, a: Union[int, float], b: Union[int, float]) -> Dict[str, Any]:
    """
    Executes a calculation based on the operation and operands supplied by the LLM.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide', 'power'
        a: First number
        b: Second number
        
    Returns:
        dict: A dictionary containing the calculation result or an error message.
    """
    op = operation.strip().lower()
    
    try:
        a_val = float(a)
        b_val = float(b)
    except (ValueError, TypeError) as err:
        return {
            "status": "error",
            "error": f"Invalid numeric input: a='{a}', b='{b}'. Details: {err}"
        }

    if op == "add":
        res = a_val + b_val
    elif op == "subtract":
        res = a_val - b_val
    elif op == "multiply":
        res = a_val * b_val
    elif op == "divide":
        if b_val == 0:
            return {
                "status": "error",
                "error": "Division by zero is undefined."
            }
        res = a_val / b_val
    elif op == "power":
        try:
            res = a_val ** b_val
        except OverflowError:
            return {
                "status": "error",
                "error": "Calculation resulted in an overflow error (number too large)."
            }
    else:
        return {
            "status": "error",
            "error": f"Unsupported operation '{operation}'. Allowed operations are: add, subtract, multiply, divide, power."
        }

    # Format result: if it's a whole number, cast to int for clean display
    formatted_res = int(res) if res.is_integer() else round(res, 6)

    return {
        "status": "success",
        "operation": op,
        "a": a_val,
        "b": b_val,
        "result": formatted_res
    }


if __name__ == "__main__":
    # Quick self-test
    print("Test add:", execute_calculator("add", 10, 5))
    print("Test divide:", execute_calculator("divide", 10, 2))
    print("Test div by zero:", execute_calculator("divide", 10, 0))
