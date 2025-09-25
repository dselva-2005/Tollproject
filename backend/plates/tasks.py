import os
import django

# Make sure Django settings are set before importing models
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")  # <-- your settings module
django.setup()

from plates.models import DetectedPlates

def update_database(result):
    """Add a plate if it does not exist and is valid."""
    # Guard against None or empty/whitespace strings
    if result is None:
        return

    plate_no = str(result).strip()
    if not plate_no:
        return

    # Check if the plate already exists
    if not DetectedPlates.objects.filter(plate_no=plate_no).exists():
        DetectedPlates.objects.create(plate_no=plate_no)

def clear_database():
    """Clear all plate records."""
    DetectedPlates.objects.all()#.delete()
