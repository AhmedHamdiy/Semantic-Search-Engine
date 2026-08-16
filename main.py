from src.vec_db import VecDB
from src.utils import DatabaseGenerator


if __name__ == "__main__":

    generator = DatabaseGenerator(db_file_path="saved_db.dat", db_size=100_000)
    generator.generate()

    db = VecDB(database_file_path="saved_db.dat", index_file_path="index.dat", new_db=False)
    all_rows = db.get_all_rows()
    print(f"Database shape: {all_rows.shape}")