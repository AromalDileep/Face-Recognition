import time
import cv2
import logging
from utils.storage import save_embeddings


class Enroller:
    def __init__(self, embeddings_db):
        self.db = embeddings_db

    def enroll(self, name, embedding):
        """
        Add a single embedding for the given name and save to disk.
        """
        self.db.setdefault(name, [])
        self.db[name].append(embedding)
        save_embeddings(self.db)
        logging.info(f"Enrolled one image for {name}")
        return True

    def remove(self, name):
        """
        Remove a profile from the database.
        """
        if name in self.db:
            del self.db[name]
            save_embeddings(self.db)
            logging.info(f"Removed profile for {name}")
            return True
        return False
