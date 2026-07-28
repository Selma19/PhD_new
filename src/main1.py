"""Creates the kernel database for all agents available
and fills its Main table.
"""
from database import Kernel_db

if __name__ == "__main__":
    db = Kernel_db()
    db.create()
    db.connect()
    db._fill_main()
