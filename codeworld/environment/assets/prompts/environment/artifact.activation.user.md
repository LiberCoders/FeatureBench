Activation:
- Return a source-able Bash script that enters /testbed and activates the exact environment created by the
  Dockerfile. If no activation is required, enter /testbed and make that no-op explicit.
- Do not use exit, exec, or return; the host sources this script into its own shell.
