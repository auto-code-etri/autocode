import os
import uuid

import docker
from flask import Flask, request

app = Flask(__name__)
client = docker.from_env()
logger = app.logger

def _execute(
    code: str,
    lang: str = "python",
    stdin: str = "",
    version: str = None,
    mem_limit: str = "128m",
    cpu_limit: int = 1,  # need cgroup support
    timeout: int = 3,
    trace: bool = False,
    **kwargs,
):
    code_id = uuid.uuid4()
    if lang == "python":
        image = f"python:3.9-slim-test"
        ext = "py"
        if trace:
            command = f"/bin/sh -c \"timeout {timeout}s /bin/sh -c 'python3 -u -m trace --trace /code.{ext} < /stdin.in; echo Exit Code: $?;' || echo 'Failed'\""
        else:
            command = f"/bin/sh -c \"timeout {timeout}s /bin/sh -c 'python3 -u /code.{ext} < /stdin.in; echo Exit Code: $?;' || echo 'Failed'\""
    elif lang == "c":
        raise NotImplementedError("C is not supported yet")
    elif lang == "cpp":
        raise NotImplementedError("C++ is not supported yet")
    elif lang == "java":
        raise NotImplementedError("Java is not supported yet")
    else:
        raise ValueError("Invalid language")
    
    try:
        client.images.get(f"python:3.9-slim-test")
    except docker.errors.ImageNotFound:
        client.images.pull(f"python:3.9-slim-test")

    # save code to a tmp file
    code_file = f"/tmp/{code_id}.{ext}"
    with open(code_file, "w") as f:
        f.write(code)
        
    # save stdin to a tmp file
    stdin_file = f"/tmp/{code_id}.in"
    with open(stdin_file, "w") as f:
        f.write(stdin)

    container_name = f"CodeExecContainer_{code_id}"
    try:
        # worker_id = os.getenv("GUNICORN_WORKER_ID", "0")
        # cpu_start = int(worker_id) * cpu_limit
        # cpu_end = cpu_start + cpu_limit - 1
        # cpuset_cpus = f"{cpu_start}-{cpu_end}" if cpu_limit > 1 else f"{cpu_start}"

        container = client.containers.run(
            image,
            command,
            name=container_name,
            detach=True,
            stderr=True,
            stdout=True,
            tty=True,
            # cpuset_cpus=cpuset_cpus,  # need cgroup support
            mem_limit=mem_limit,
            volumes={
                code_file: {"bind": f"/code.{ext}", "mode": "ro"},
                stdin_file: {"bind": "/stdin.in", "mode": "ro"},
            },
            environment={"PYTHONUNBUFFERED": "1"},
        )
        container.wait()
        response = container.logs().decode("utf-8")
        container.remove()

        os.remove(code_file)
        os.remove(stdin_file)
        return response

    except docker.errors.ContainerError as e:
        os.remove(code_file)
        os.remove(stdin_file)
        return 'Failed'
    except Exception as e:
        os.remove(code_file)
        os.remove(stdin_file)
        return 'Failed'


@app.route("/execute", methods=["POST"])
def execute():
    try:
        logger.info(f"Request: {request.json}")
        response = _execute(**request.json)
        logger.info(f"Response: {response}")
        return {"output": response}
    except ValueError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5097)
