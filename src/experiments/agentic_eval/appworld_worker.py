"""One isolated AppWorld per process; JSON-lines control protocol on stdin/stdout."""
import contextlib
import json
import sys
import traceback


def main():
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from appworld import AppWorld
        world = AppWorld(task_id=sys.argv[1], experiment_name=sys.argv[2],
                         ground_truth_mode="full" if "--oracle" in sys.argv else "minimal",
                         raise_on_failure=False, timeout_seconds=60)
    def send(value):
        protocol.write(json.dumps(value, default=str) + "\n")
        protocol.flush()
    send({"instruction": world.task.instruction, "supervisor": dict(world.task.supervisor)})
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    if request["method"] == "execute":
                        result = {"output": world.execute(request["code"]), "completed": world.task_completed()}
                    elif request["method"] == "evaluate":
                        tracker = world.evaluate()
                        result = {"success": tracker.success}
                    elif request["method"] == "oracle":
                        if "--oracle" not in sys.argv:
                            raise ValueError("Oracle access disabled")
                        world.execute(world.task.ground_truth.compiled_solution_code + "\nsolution(apis, requester)")
                        result = {"success": world.evaluate().success}
                    else:
                        raise ValueError("Unknown method")
                send(result)
            except Exception:
                send({"error": traceback.format_exc()})
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            world.close()


if __name__ == "__main__":
    main()
