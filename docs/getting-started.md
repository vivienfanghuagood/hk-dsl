# Getting started

From nothing to a kernel that is right and at least as fast as
`torch.compile`, then to one you wrote yourself, in three pages:

1. [Install](install.md): what you need (hipcc, torch, a supported GPU), the
   tested toolchains, the wheel from PyPI or a build from source, and a clone
   for the examples and tests.
2. [Quickstart](quickstart.md): a kernel from a template, `hk.check` on it,
   a kernel written by hand, the four entry points, timing, and your model in
   vLLM -- with what differs between a W7900D and an MI355X.
3. [Examples](examples.md): fifteen examples, from a vector add to flash
   attention, which GPUs each runs on, and `validate.py`.

Which GPUs run what: [supported hardware](supported-hardware.md); which hipcc:
[the toolchain matrix](toolchains.md).
