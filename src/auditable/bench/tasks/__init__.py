"""Importing this package registers every board task. One module per task family."""
from auditable.bench.tasks import detect  # noqa: F401
from auditable.bench.tasks import live  # noqa: F401
from auditable.bench.tasks import localize  # noqa: F401
from auditable.bench.tasks import pre  # noqa: F401
