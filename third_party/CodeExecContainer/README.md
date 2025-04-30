# CodeExecContainer

This project aims to create a containerized environment for executing code snippets. It provides a secure and isolated environment for running code, making it suitable for online coding platforms, code editors, and other similar applications.

## Features

- Containerized execution environment
- Support for multiple programming languages
- Secure sandboxing to prevent unauthorized access
- Resource limitations to prevent abuse
- Easy integration with existing applications

## Installation

To install and run the CodeExecContainer, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/CodeExecContainer.git
   ```

2. Navigate to the project directory:

   ```shell
   cd CodeExecContainer
   ```

3. Install the dependencies:

   ```shell
   pip install -r requirements.txt
   ```

4. Start the application:

   ```shell
   source run.sh --port 5097
   ```

## Usage

To use the CodeExecContainer, follow these steps:

1.  Make a POST request to the `/execute` endpoint with the following parameters:

    - `code`: The code snippet to be executed.
    - `lang`: The programming language of the code snippet.
    - `version`: The version of the programming language (optional).
    - `stdin`: The standard input for the code snippet (optional).
    - `timeout`: The maximum execution time for the code snippet in seconds (optional).
    - `memory_limit`: The maximum memory usage for the code snippet in bytes (optional).
    - `cpu_limit`: The maximum CPU usage for the code snippet in seconds (optional).

    Example using cURL:

    ```shell
    curl -X POST -H "Content-Type: application/json" \
        -d '{"code": "print(\"Hello, World!\")", "lang": "python"}' \
        http://localhost:5097/execute
    ```

2.  The response will contain the output of the executed code.

    Example response:

    ```json
    {
      "output": "Hello, World!\n"
    }
    ```

3.  If an error occurs during execution, the response will contain an error message.

        Example error response:

        ```json
        {
            "error": "Execution failed: TimeoutError: Execution timed out",
        }
        ```

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
