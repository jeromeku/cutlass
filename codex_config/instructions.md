# Global operating rules for Codex

**Non-negotiable logging policy**

- For every session:
  1. Determine the repo root with `git rev-parse --show-toplevel`; if not a git repo, use the current directory.
  2. Create (if missing) `<repo-root>/codex-logs/`.
  3. Write a Markdown transcript to `codex-logs/<YYYYMMDD-HHMMSS>-session.md` that includes:
     - Title line: repo name (or directory), model, profile, sandbox mode.
     - All user messages and Codex replies, verbatim (fenced Markdown blocks).
     - A **Commands** section listing every command you ran with start time, exit code, and duration.
     - A **Changes** section summarizing files you created/edited/removed and the current `git diff --no-ext-diff`.
  4. Update/append a machine-readable provenance file at `codex-logs/provenance.jsonl` with JSON lines for each command run: `{ts, cwd, cmd, exit_code, duration_ms}`.
- If no files were changed or no commands were executed, state that explicitly in the log.
- When in doubt, **prefer writing the log first** and then continuing the task so the transcript is never lost.
