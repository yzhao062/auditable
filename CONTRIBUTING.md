# Contributing to auditable

Thanks for your interest. `auditable` is early, and contributions are welcome.

## Developer Certificate of Origin (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org/). Sign off every commit with `git commit -s`, which adds a `Signed-off-by` line certifying you wrote the change or have the right to submit it under the project's Apache-2.0 license. Pull requests without sign-off cannot be merged.

The DCO keeps contribution ownership clean, so the project can relicense or offer a hosted version later without tracking down past contributors.

## Development Setup

```bash
git clone https://github.com/yzhao062/auditable
cd auditable
pip install -e ".[dev]"
pytest -q
```

## Before You Open a Pull Request

- Run `pytest -q` and keep it green.
- Keep the public API small; new surface area needs a clear reason.
- One focused change per pull request.
