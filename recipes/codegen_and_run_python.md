# Recipe: Codegen + Run (Python) in sandbox

Goal: Generate a Python file in the sandbox, run it, and report results.

## Tools available
- write_file(path, content): write UTF-8 file into sandbox.
- run_cli(command): run restricted CLI in sandbox (NO pipes/redirects/chaining).
- read_file(path): read files.
- list_dir(path): list directory.
- final(text): MUST be used to output the final answer.

## Policy / discipline
- Never guess outputs. Use tools.
- If any tool returns ok=false, you MUST final('TOOL_FAIL').
- Keep final output minimal and exactly in the format the user requested.

## Strategy
1) Decide the filename (prefer short: main.py / hello.py / task.py).
2) Use write_file to create the Python script.
3) If the user asked to execute it:
   - Try to run it using run_cli with an allowed command.
   - If `python3` is not available as an allowed CLI command, do NOT attempt to bypass restrictions.
   - Instead, return TOOL_FAIL (or explain in minimal form if asked).
4) If execution is not possible, still return the file path/name if the user only requested generation.

## Output templates
- If user asked "only return filename": final with just `task.py`.
- If user asked "return stdout only": final with stdout text.
- If user asked "return both": final with `filename\n<stdout>`.
