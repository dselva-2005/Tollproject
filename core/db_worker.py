import psycopg2  # or pymongo for MongoDB
import os
import sys
import django

# 1. Add backend folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# 2. Set Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")
django.setup()

# 3. Import your ORM models
from plates.models import Plate  # adjust 'myapp'

def update_database(result):
    plate = Plate(plate_no = result)
    plate.save()

def clear_database():
    print(Plate.objects.all())
