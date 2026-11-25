import os, sys, time, subprocess
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from django.conf import settings
from django.db import connection

# Robust DNS fallback for Windows environments where local DNS may lag
host = settings.DATABASES['default'].get('HOST', '')
if 'neon.tech' in host:
    try:
        import socket
        socket.gethostbyname(host)
    except socket.gaierror:
        try:
            res = subprocess.run(
                ['powershell', '-Command', f'Resolve-DnsName -Server 8.8.8.8 {host} -Type A | Select-Object -ExpandProperty IP4Address'],
                capture_output=True, text=True, timeout=5
            )
            ips = res.stdout.strip().split()
            if ips:
                options = settings.DATABASES['default'].setdefault('OPTIONS', {})
                options['hostaddr'] = ips[0]
        except Exception:
            pass

from rag.models import LegalSourceDocument, LegalProvisionChunk, StatuteAmendmentRecord

for attempt in range(1, 8):
    try:
        connection.close()
        connection.ensure_connection()
        doc_count = LegalSourceDocument.objects.count()
        chunk_count = LegalProvisionChunk.objects.count()
        amend_count = StatuteAmendmentRecord.objects.count()
        print('=== LIVE NEON PRODUCTION ROW COUNTS ===')
        print(f'LegalSourceDocument:    {doc_count}')
        print(f'LegalProvisionChunk:     {chunk_count}')
        print(f'StatuteAmendmentRecord: {amend_count}')
        assert doc_count == 0, 'Found uncleaned documents!'
        assert chunk_count == 0, 'Found uncleaned chunks!'
        assert amend_count == 0, 'Found uncleaned amendments!'
        print('ZERO RECORDS CONFIRMED.')
        sys.exit(0)
    except Exception as e:
        if attempt == 7:
            print(f'Final attempt failed: {e}')
            sys.exit(1)
        print(f'Attempt {attempt} connection lag ({e}). Retrying in 2s...')
        time.sleep(2)
