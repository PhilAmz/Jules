# In tools.py
import numexpr # Using numexpr for safer evaluation

def calculate(expression: str) -> str:
    try:
        # For safety, ensure only basic math operations are allowed if not using a safe parser
        # This is a simple check; a proper allowlist would be more robust
        # numexpr itself is safer than eval(), but this adds an extra layer for demonstration
        allowed_chars = "0123456789+-*/(). "
        # Allow 'e' or 'E' for scientific notation if numexpr handles it, common in its inputs
        # For simplicity, we'll stick to the provided allowed_chars for now.
        # If numexpr needs 'e' for its own syntax (e.g. 1e5), it might be an issue.
        # However, the prompt's allowed_chars doesn't include 'e'.

        for char_in_expr in expression:
            if char_in_expr not in allowed_chars:
                return "Error: Invalid characters in expression. Only numbers and basic math operators (+, -, *, /) are allowed."

        # Using numexpr for safer evaluation than eval()
        # numexpr.evaluate() can take dicts for local_dict and global_dict
        result = numexpr.evaluate(expression, local_dict={}, global_dict={}).item() # .item() to get scalar from 0-d array
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except SyntaxError: # This might not be directly raised by numexpr if the char check catches it.
        return "Error: Invalid syntax in mathematical expression."
    except Exception as e:
        # numexpr might raise its own specific errors for parsing or evaluation not caught by SyntaxError
        return f"Error during calculation: {str(e)}"

def search_web(query: str) -> dict: # Changed return type
    query_lower = query.lower()
    if "chainlit documentation" in query_lower:
        return {"type": "url", "content": "https://docs.chainlit.io", "description": "Official Chainlit Documentation"}
    elif "what is chainlit" in query_lower:
        return {"type": "text", "content": "Chainlit is an open-source Python package that makes it incredibly fast to build Chat GPT like applications with your own business logic and data."}
    elif "show me a logo of python" in query_lower: # Example for an image
            # Using a known, publicly accessible image URL
        return {"type": "image_url", "content": "https://www.python.org/static/community_logos/python-logo-master-v3-TM.png", "description": "Python Logo"}
    else:
        return {"type": "text", "content": f"Mock search result for '{query}': No specific information found, but you can try searching on your preferred search engine."}

import io
import sys
import traceback # For capturing tracebacks from exec

def execute_python_code(code_string: str) -> str:
    """
    Executes a string of Python code and captures its stdout, stderr, and exceptions.

    !!! WARNING !!!
    This function uses exec() to execute arbitrary Python code.
    This is a MAJOR SECURITY RISK if the code_string is not from a trusted source.
    In a production environment, code execution should be handled in a
    secure, sandboxed environment (e.g., using Docker containers, Kata containers,
    RestrictedPython, or a dedicated microservice).
    For this example, we proceed with exec() for simplicity, assuming the LLM
    provides benign code or the user understands the risk in a controlled demo.
    !!! WARNING !!!
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_stdout = io.StringIO()
    redirected_stderr = io.StringIO()
    sys.stdout = redirected_stdout
    sys.stderr = redirected_stderr

    output_parts = []

    try:
        # Execute in a new, empty global scope for some isolation
        exec(code_string, {})

        stdout_val = redirected_stdout.getvalue()
        stderr_val = redirected_stderr.getvalue()

        if stdout_val:
            output_parts.append(f"Stdout:\n{stdout_val}")
        if stderr_val:
            output_parts.append(f"Stderr:\n{stderr_val}")

        if not stdout_val and not stderr_val:
            output_parts.append("Code executed successfully with no explicit print output or errors.")

    except Exception as e:
        # Capture the full traceback
        tb_str = traceback.format_exc()
        output_parts.append(f"Exception during execution:\n{tb_str}")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return "\n---\n".join(output_parts) if output_parts else "Code execution attempt finished."
