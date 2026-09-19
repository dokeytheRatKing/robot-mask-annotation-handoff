"""Start the local mask editor and open its browser UI (stdlib only)."""
import sys
from audit_server import main

if __name__=='__main__':
    if '--open-browser' not in sys.argv:sys.argv.append('--open-browser')
    main()
