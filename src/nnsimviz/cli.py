"""Console entry point: launch the Streamlit GUI.

Installed as the `nnsimviz` command (see pyproject.toml). It simply invokes
`streamlit run` on the bundled app module, so users do not need to know the
installed file path.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit app via the streamlit CLI."""
    from streamlit.web import cli as stcli

    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
