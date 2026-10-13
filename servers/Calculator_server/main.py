from fastmcp import FastMCP

# Create an MCP server instance
mcp = FastMCP("Calculator MCP Server")

# ---- Define the tools ----

@mcp.tool
def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b

@mcp.tool
def subtract(a: float, b: float) -> float:
    """Return the difference between two numbers."""
    return a - b

@mcp.tool
def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b

@mcp.tool
def divide(a: float, b: float) -> float:
    """Return the quotient of two numbers. Raises an error if dividing by zero."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

@mcp.tool
def power(a: float, b: float) -> float:
    """Return a raised to the power of b."""
    return a ** b

# ---- Run the MCP server ----
if __name__ == "__main__":
    mcp.run()

