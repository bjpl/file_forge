"""TDD Tests for FileForge database module.

RED phase: Tests written first, defining expected behavior.
These tests will FAIL until implementation is complete.

Test Coverage:
- Database initialization and schema creation
- File CRUD operations (create, read, update, delete)
- Operation journal/audit trail
- Query operations with filtering
- Object detection storage
- Face detection and clustering
- NSFW detection results
- Processing error tracking
- Database statistics and analytics
- Concurrent access (WAL mode)
"""
import pytest
from pathlib import Path
from datetime import datetime
import json
import numpy as np


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for database files."""
    return tmp_path


@pytest.fixture
def memory_db():
    """Provide in-memory database for fast testing."""
    from fileforge.storage.database import Database
    db = Database(":memory:")
    db.initialize()
    return db


class TestDatabaseInitialization:
    """Tests for database initialization and schema creation."""

    def test_creates_database_file(self, temp_dir):
        """Database should create file at specified path."""
        from fileforge.storage.database import Database
        db_path = temp_dir / "test.db"
        db = Database(db_path)
        db.initialize()
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    def test_creates_all_required_tables(self, temp_dir):
        """Database should create all required tables on init."""
        from fileforge.storage.database import Database
        db = Database(temp_dir / "test.db")
        db.initialize()

        tables = db.get_tables()
        required = ['files', 'operations', 'detected_objects', 'extracted_text',
                   'faces', 'nsfw_detections', 'processing_errors', 'processing_runs']
        for table in required:
            assert table in tables, f"Missing table: {table}"

    def test_creates_proper_indexes(self, temp_dir):
        """Database should create indexes for performance."""
        from fileforge.storage.database import Database
        db = Database(temp_dir / "test.db")
        db.initialize()

        indexes = db.get_indexes()
        # Should have indexes on file_hash, file_type, batch_id, etc.
        assert len(indexes) > 0

    def test_enables_wal_mode_by_default(self, temp_dir):
        """WAL mode should be enabled for concurrent access."""
        from fileforge.storage.database import Database
        db = Database(temp_dir / "test.db", wal_mode=True)
        db.initialize()
        assert db.get_journal_mode() == "wal"

    def test_disables_wal_mode_when_requested(self, temp_dir):
        """Should support disabling WAL mode."""
        from fileforge.storage.database import Database
        db = Database(temp_dir / "test.db", wal_mode=False)
        db.initialize()
        assert db.get_journal_mode() in ["delete", "persist"]

    def test_supports_in_memory_database(self):
        """Should support in-memory database for testing."""
        from fileforge.storage.database import Database
        db = Database(":memory:")
        db.initialize()
        assert db.get_tables()  # Should have tables

    def test_handles_existing_database_gracefully(self, temp_dir):
        """Should not fail when initializing existing database."""
        from fileforge.storage.database import Database
        db_path = temp_dir / "test.db"

        db1 = Database(db_path)
        db1.initialize()
        db1.close()  # Close first connection before opening second

        # Second initialization should not fail
        db2 = Database(db_path)
        db2.initialize()
        db2.close()

        assert db_path.exists()

    def test_sets_pragmas_for_performance(self, temp_dir):
        """Should set appropriate SQLite pragmas."""
        from fileforge.storage.database import Database
        db = Database(temp_dir / "test.db")
        db.initialize()

        # Check cache size is set
        cache_size = db.get_pragma("cache_size")
        assert cache_size is not None

        # Check foreign keys are enabled
        foreign_keys = db.get_pragma("foreign_keys")
        assert foreign_keys == 1


class TestFileOperations:
    """Tests for file CRUD operations."""

    def test_upsert_file_creates_new_record(self, memory_db):
        """Upserting new file should create record."""
        file_data = {
            'file_path': '/test/file.txt',
            'file_hash': 'abc123',
            'file_type': 'text',
            'original_name': 'file.txt',
            'content_text': 'Test content',
            'confidence': 0.95
        }

        file_id = memory_db.upsert_file(file_data)
        assert file_id is not None
        assert file_id > 0

    def test_upsert_file_updates_existing_record(self, memory_db):
        """Upserting existing file should update record."""
        file_data = {
            'file_path': '/test/file.txt',
            'file_hash': 'abc123',
            'file_type': 'text',
            'original_name': 'file.txt'
        }

        id1 = memory_db.upsert_file(file_data)
        file_data['content_text'] = 'Updated content'
        id2 = memory_db.upsert_file(file_data)

        assert id1 == id2  # Same record updated

        retrieved = memory_db.get_file('/test/file.txt')
        assert retrieved['content_text'] == 'Updated content'

    def test_upsert_file_updates_timestamp(self, memory_db):
        """Updating file should update updated_at timestamp."""
        file_data = {
            'file_path': '/test/file.txt',
            'file_hash': 'abc123',
            'file_type': 'text'
        }

        memory_db.upsert_file(file_data)
        first_updated = memory_db.get_file('/test/file.txt')['updated_at']

        # Wait a moment and update
        import time
        time.sleep(0.1)

        file_data['content_text'] = 'New content'
        memory_db.upsert_file(file_data)
        second_updated = memory_db.get_file('/test/file.txt')['updated_at']

        assert second_updated > first_updated

    def test_get_file_by_path(self, memory_db):
        """Should retrieve file by path."""
        memory_db.upsert_file({
            'file_path': '/test/file.txt',
            'file_hash': 'abc123',
            'file_type': 'text',
            'original_name': 'file.txt'
        })

        result = memory_db.get_file('/test/file.txt')
        assert result is not None
        assert result['file_hash'] == 'abc123'
        assert result['original_name'] == 'file.txt'

    def test_get_file_by_hash(self, memory_db):
        """Should retrieve file by hash."""
        memory_db.upsert_file({
            'file_path': '/test/file.txt',
            'file_hash': 'unique_hash_123',
            'file_type': 'text'
        })

        result = memory_db.get_file_by_hash('unique_hash_123')
        assert result is not None
        assert result['file_path'] == '/test/file.txt'

    def test_get_file_by_id(self, memory_db):
        """Should retrieve file by ID."""
        file_id = memory_db.upsert_file({
            'file_path': '/test/file.txt',
            'file_hash': 'abc123',
            'file_type': 'text'
        })

        result = memory_db.get_file_by_id(file_id)
        assert result is not None
        assert result['file_id'] == file_id

    def test_get_nonexistent_file_returns_none(self, memory_db):
        """Getting nonexistent file should return None."""
        assert memory_db.get_file('/nonexistent/file.txt') is None
        assert memory_db.get_file_by_hash('nonexistent_hash') is None
        assert memory_db.get_file_by_id(99999) is None

    def test_delete_file(self, memory_db):
        """Should delete file record."""
        file_id = memory_db.upsert_file({
            'file_path': '/test/file.txt',
            'file_hash': 'abc123',
            'file_type': 'text'
        })

        memory_db.delete_file(file_id)
        assert memory_db.get_file_by_id(file_id) is None

    def test_upsert_file_with_json_fields(self, memory_db):
        """Should handle JSON fields properly."""
        file_data = {
            'file_path': '/test/file.txt',
            'file_hash': 'abc123',
            'file_type': 'text',
            'tags': json.dumps(['important', 'finance']),
            'metadata': json.dumps({'author': 'John Doe', 'version': '1.0'})
        }

        file_id = memory_db.upsert_file(file_data)
        result = memory_db.get_file_by_id(file_id)

        tags = json.loads(result['tags'])
        assert 'important' in tags

        metadata = json.loads(result['metadata'])
        assert metadata['author'] == 'John Doe'


class TestOperationJournal:
    """Tests for operation audit trail."""

    def test_add_operation_records_action(self, memory_db):
        """Should record operation in journal."""
        op_id = memory_db.add_operation({
            'batch_id': 'batch_001',
            'operation_type': 'rename',
            'source_path': '/old/name.txt',
            'dest_path': '/new/name.txt',
            'old_value': 'name.txt',
            'new_value': 'renamed.txt'
        })

        assert op_id is not None
        assert op_id > 0

    def test_add_operation_sets_timestamp(self, memory_db):
        """Should set created_at timestamp automatically."""
        op_id = memory_db.add_operation({
            'batch_id': 'batch_001',
            'operation_type': 'rename',
            'source_path': '/test.txt'
        })

        op = memory_db.get_operation(op_id)
        assert op['created_at'] is not None
        assert isinstance(op['created_at'], str)  # ISO format timestamp

    def test_get_operations_by_batch(self, memory_db):
        """Should retrieve all operations in a batch."""
        memory_db.add_operation({'batch_id': 'batch_001', 'operation_type': 'rename', 'source_path': '/a.txt'})
        memory_db.add_operation({'batch_id': 'batch_001', 'operation_type': 'move', 'source_path': '/b.txt'})
        memory_db.add_operation({'batch_id': 'batch_002', 'operation_type': 'rename', 'source_path': '/c.txt'})

        batch_ops = memory_db.get_operations_by_batch('batch_001')
        assert len(batch_ops) == 2
        assert all(op['batch_id'] == 'batch_001' for op in batch_ops)

    def test_get_operations_by_type(self, memory_db):
        """Should filter operations by type."""
        memory_db.add_operation({'batch_id': 'b1', 'operation_type': 'rename', 'source_path': '/a.txt'})
        memory_db.add_operation({'batch_id': 'b1', 'operation_type': 'move', 'source_path': '/b.txt'})
        memory_db.add_operation({'batch_id': 'b1', 'operation_type': 'rename', 'source_path': '/c.txt'})

        rename_ops = memory_db.get_operations_by_type('rename')
        assert len(rename_ops) == 2

    def test_mark_operation_undone(self, memory_db):
        """Should mark operation as undone."""
        op_id = memory_db.add_operation({
            'batch_id': 'batch_001',
            'operation_type': 'rename',
            'source_path': '/test.txt'
        })

        memory_db.mark_operation_undone(op_id)

        op = memory_db.get_operation(op_id)
        assert op['undone'] is True
        assert op['undone_at'] is not None

    def test_get_undoable_operations(self, memory_db):
        """Should retrieve operations that can be undone."""
        op1_id = memory_db.add_operation({'batch_id': 'b1', 'operation_type': 'rename', 'source_path': '/a.txt'})
        op2_id = memory_db.add_operation({'batch_id': 'b1', 'operation_type': 'move', 'source_path': '/b.txt'})

        memory_db.mark_operation_undone(op1_id)

        undoable = memory_db.get_undoable_operations('b1')
        assert len(undoable) == 1
        assert undoable[0]['operation_id'] == op2_id


class TestQueryOperations:
    """Tests for file querying and filtering."""

    def test_query_files_by_type(self, memory_db):
        """Should filter files by type."""
        memory_db.upsert_file({'file_path': '/a.txt', 'file_hash': 'h1', 'file_type': 'text'})
        memory_db.upsert_file({'file_path': '/b.jpg', 'file_hash': 'h2', 'file_type': 'image'})
        memory_db.upsert_file({'file_path': '/c.txt', 'file_hash': 'h3', 'file_type': 'text'})

        results = memory_db.query_files({'file_type': 'text'})
        assert len(results) == 2
        assert all(f['file_type'] == 'text' for f in results)

    def test_query_files_by_category(self, memory_db):
        """Should filter files by category."""
        memory_db.upsert_file({'file_path': '/a.txt', 'file_hash': 'h1', 'file_type': 'text', 'category': 'invoices'})
        memory_db.upsert_file({'file_path': '/b.txt', 'file_hash': 'h2', 'file_type': 'text', 'category': 'receipts'})
        memory_db.upsert_file({'file_path': '/c.txt', 'file_hash': 'h3', 'file_type': 'text', 'category': 'invoices'})

        results = memory_db.query_files({'category': 'invoices'})
        assert len(results) == 2
        assert all(f['category'] == 'invoices' for f in results)

    def test_query_files_with_text_search(self, memory_db):
        """Should search within content_text."""
        memory_db.upsert_file({'file_path': '/a.txt', 'file_hash': 'h1', 'file_type': 'text',
                               'content_text': 'Invoice for consulting services'})
        memory_db.upsert_file({'file_path': '/b.txt', 'file_hash': 'h2', 'file_type': 'text',
                               'content_text': 'Meeting notes from Monday'})
        memory_db.upsert_file({'file_path': '/c.txt', 'file_hash': 'h3', 'file_type': 'text',
                               'content_text': 'Invoice for product purchase'})

        results = memory_db.query_files({'text_search': 'invoice'})
        assert len(results) == 2
        assert all('invoice' in f['content_text'].lower() for f in results)

    def test_query_files_by_tags(self, memory_db):
        """Should filter files by tags."""
        memory_db.upsert_file({'file_path': '/a.txt', 'file_hash': 'h1', 'file_type': 'text',
                               'tags': json.dumps(['finance', 'important'])})
        memory_db.upsert_file({'file_path': '/b.txt', 'file_hash': 'h2', 'file_type': 'text',
                               'tags': json.dumps(['personal'])})
        memory_db.upsert_file({'file_path': '/c.txt', 'file_hash': 'h3', 'file_type': 'text',
                               'tags': json.dumps(['finance', 'receipts'])})

        results = memory_db.query_files({'tag': 'finance'})
        assert len(results) == 2

    def test_query_files_with_multiple_filters(self, memory_db):
        """Should combine multiple filter criteria."""
        memory_db.upsert_file({'file_path': '/a.txt', 'file_hash': 'h1', 'file_type': 'text',
                               'category': 'invoices', 'content_text': 'Invoice 2024'})
        memory_db.upsert_file({'file_path': '/b.jpg', 'file_hash': 'h2', 'file_type': 'image',
                               'category': 'invoices'})
        memory_db.upsert_file({'file_path': '/c.txt', 'file_hash': 'h3', 'file_type': 'text',
                               'category': 'receipts', 'content_text': 'Receipt 2024'})

        results = memory_db.query_files({
            'file_type': 'text',
            'category': 'invoices'
        })
        assert len(results) == 1
        assert results[0]['file_path'] == '/a.txt'

    def test_query_files_with_limit(self, memory_db):
        """Should respect limit parameter."""
        for i in range(10):
            memory_db.upsert_file({'file_path': f'/file{i}.txt', 'file_hash': f'h{i}', 'file_type': 'text'})

        results = memory_db.query_files({}, limit=5)
        assert len(results) == 5

    def test_query_files_with_offset(self, memory_db):
        """Should support pagination with offset."""
        for i in range(10):
            memory_db.upsert_file({'file_path': f'/file{i}.txt', 'file_hash': f'h{i}', 'file_type': 'text'})

        page1 = memory_db.query_files({}, limit=5, offset=0)
        page2 = memory_db.query_files({}, limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0]['file_path'] != page2[0]['file_path']


class TestDetectedObjects:
    """Tests for object detection storage."""

    def test_add_detected_objects(self, memory_db):
        """Should store detected objects for a file."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})

        memory_db.add_detected_objects(file_id, [
            {'label': 'person', 'confidence': 0.95, 'bbox': [10, 20, 100, 200]},
            {'label': 'dog', 'confidence': 0.87, 'bbox': [150, 100, 200, 180]}
        ])

        objects = memory_db.get_detected_objects(file_id)
        assert len(objects) == 2
        assert objects[0]['label'] == 'person'
        assert objects[0]['confidence'] == 0.95

    def test_detected_objects_stores_bbox_as_json(self, memory_db):
        """Bounding box should be stored as JSON."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})

        bbox = [10, 20, 100, 200]
        memory_db.add_detected_objects(file_id, [
            {'label': 'person', 'confidence': 0.95, 'bbox': bbox}
        ])

        objects = memory_db.get_detected_objects(file_id)
        retrieved_bbox = json.loads(objects[0]['bbox']) if isinstance(objects[0]['bbox'], str) else objects[0]['bbox']
        assert retrieved_bbox == bbox

    def test_query_files_by_detected_object(self, memory_db):
        """Should find files containing specific objects."""
        file_id1 = memory_db.upsert_file({'file_path': '/img1.jpg', 'file_hash': 'h1', 'file_type': 'image'})
        file_id2 = memory_db.upsert_file({'file_path': '/img2.jpg', 'file_hash': 'h2', 'file_type': 'image'})

        memory_db.add_detected_objects(file_id1, [{'label': 'dog', 'confidence': 0.9}])
        memory_db.add_detected_objects(file_id2, [{'label': 'cat', 'confidence': 0.9}])

        results = memory_db.query_files_by_object('dog')
        assert len(results) == 1
        assert results[0]['file_path'] == '/img1.jpg'

    def test_delete_detected_objects_on_file_delete(self, memory_db):
        """Should cascade delete detected objects when file is deleted."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})
        memory_db.add_detected_objects(file_id, [{'label': 'person', 'confidence': 0.95}])

        memory_db.delete_file(file_id)

        objects = memory_db.get_detected_objects(file_id)
        assert len(objects) == 0


class TestFaceStorage:
    """Tests for face detection and clustering storage."""

    def test_add_face_with_embedding(self, memory_db):
        """Should store face with embedding vector."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})

        embedding = np.random.rand(512).astype(np.float32)
        face_id = memory_db.add_face(file_id, {
            'embedding': embedding,
            'confidence': 0.98,
            'bbox': [50, 50, 150, 200]
        })

        assert face_id is not None
        assert face_id > 0

    def test_get_face_embedding(self, memory_db):
        """Should retrieve face embedding as numpy array."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})

        embedding = np.random.rand(512).astype(np.float32)
        face_id = memory_db.add_face(file_id, {
            'embedding': embedding,
            'confidence': 0.98
        })

        face = memory_db.get_face(face_id)
        retrieved_embedding = memory_db.get_face_embedding(face_id)

        assert isinstance(retrieved_embedding, np.ndarray)
        assert retrieved_embedding.shape == (512,)
        assert np.allclose(retrieved_embedding, embedding)

    def test_update_face_cluster(self, memory_db):
        """Should update face cluster assignment."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})
        face_id = memory_db.add_face(file_id, {
            'embedding': np.random.rand(512).astype(np.float32),
            'confidence': 0.98
        })

        memory_db.update_face_cluster(face_id, cluster_id=5, cluster_name="John Doe")

        face = memory_db.get_face(face_id)
        assert face['cluster_id'] == 5
        assert face['cluster_name'] == "John Doe"

    def test_get_faces_by_cluster(self, memory_db):
        """Should retrieve all faces in a cluster."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})

        face1_id = memory_db.add_face(file_id, {'embedding': np.random.rand(512).astype(np.float32), 'confidence': 0.98})
        face2_id = memory_db.add_face(file_id, {'embedding': np.random.rand(512).astype(np.float32), 'confidence': 0.95})
        face3_id = memory_db.add_face(file_id, {'embedding': np.random.rand(512).astype(np.float32), 'confidence': 0.97})

        memory_db.update_face_cluster(face1_id, cluster_id=1, cluster_name="Person A")
        memory_db.update_face_cluster(face2_id, cluster_id=1, cluster_name="Person A")
        memory_db.update_face_cluster(face3_id, cluster_id=2, cluster_name="Person B")

        cluster1_faces = memory_db.get_faces_by_cluster(1)
        assert len(cluster1_faces) == 2

    def test_get_all_face_embeddings(self, memory_db):
        """Should retrieve all face embeddings for clustering."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})

        for i in range(5):
            memory_db.add_face(file_id, {
                'embedding': np.random.rand(512).astype(np.float32),
                'confidence': 0.9 + i * 0.01
            })

        embeddings = memory_db.get_all_face_embeddings()
        assert len(embeddings) == 5
        assert all(isinstance(emb['embedding'], np.ndarray) for emb in embeddings)


class TestNSFWDetection:
    """Tests for NSFW detection results storage."""

    def test_add_nsfw_detection(self, memory_db):
        """Should store NSFW detection results."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})

        nsfw_id = memory_db.add_nsfw_detection(file_id, {
            'is_nsfw': False,
            'confidence': 0.95,
            'scores': json.dumps({'safe': 0.95, 'nsfw': 0.05})
        })

        assert nsfw_id is not None

    def test_get_nsfw_detection(self, memory_db):
        """Should retrieve NSFW detection results."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})

        memory_db.add_nsfw_detection(file_id, {
            'is_nsfw': True,
            'confidence': 0.87,
            'scores': json.dumps({'safe': 0.13, 'nsfw': 0.87})
        })

        result = memory_db.get_nsfw_detection(file_id)
        assert result['is_nsfw'] is True
        assert result['confidence'] == 0.87

    def test_query_nsfw_files(self, memory_db):
        """Should filter files by NSFW status."""
        file_id1 = memory_db.upsert_file({'file_path': '/safe.jpg', 'file_hash': 'h1', 'file_type': 'image'})
        file_id2 = memory_db.upsert_file({'file_path': '/nsfw.jpg', 'file_hash': 'h2', 'file_type': 'image'})

        memory_db.add_nsfw_detection(file_id1, {'is_nsfw': False, 'confidence': 0.95})
        memory_db.add_nsfw_detection(file_id2, {'is_nsfw': True, 'confidence': 0.89})

        nsfw_files = memory_db.query_nsfw_files(is_nsfw=True)
        assert len(nsfw_files) == 1
        assert nsfw_files[0]['file_path'] == '/nsfw.jpg'


class TestProcessingErrors:
    """Tests for processing error tracking."""

    def test_add_processing_error(self, memory_db):
        """Should record processing errors."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})

        error_id = memory_db.add_processing_error(file_id, {
            'stage': 'object_detection',
            'error_type': 'ModelLoadError',
            'error_message': 'Failed to load YOLO model',
            'traceback': 'Stack trace here...'
        })

        assert error_id is not None

    def test_get_processing_errors_for_file(self, memory_db):
        """Should retrieve all errors for a file."""
        file_id = memory_db.upsert_file({'file_path': '/img.jpg', 'file_hash': 'h1', 'file_type': 'image'})

        memory_db.add_processing_error(file_id, {'stage': 'ocr', 'error_message': 'OCR failed'})
        memory_db.add_processing_error(file_id, {'stage': 'face_detection', 'error_message': 'No faces detected'})

        errors = memory_db.get_processing_errors(file_id)
        assert len(errors) == 2

    def test_get_processing_errors_by_stage(self, memory_db):
        """Should filter errors by processing stage."""
        file_id1 = memory_db.upsert_file({'file_path': '/img1.jpg', 'file_hash': 'h1', 'file_type': 'image'})
        file_id2 = memory_db.upsert_file({'file_path': '/img2.jpg', 'file_hash': 'h2', 'file_type': 'image'})

        memory_db.add_processing_error(file_id1, {'stage': 'ocr', 'error_message': 'OCR failed'})
        memory_db.add_processing_error(file_id2, {'stage': 'face_detection', 'error_message': 'Error'})

        ocr_errors = memory_db.get_errors_by_stage('ocr')
        assert len(ocr_errors) == 1


class TestProcessingRuns:
    """Tests for processing run tracking."""

    def test_create_processing_run(self, memory_db):
        """Should create processing run record."""
        run_id = memory_db.create_processing_run({
            'run_type': 'full_scan',
            'parameters': json.dumps({'batch_size': 100, 'enable_gpu': True})
        })

        assert run_id is not None

    def test_update_processing_run(self, memory_db):
        """Should update processing run status."""
        run_id = memory_db.create_processing_run({'run_type': 'full_scan'})

        memory_db.update_processing_run(run_id, {
            'status': 'completed',
            'files_processed': 150,
            'files_succeeded': 145,
            'files_failed': 5
        })

        run = memory_db.get_processing_run(run_id)
        assert run['status'] == 'completed'
        assert run['files_processed'] == 150

    def test_get_recent_processing_runs(self, memory_db):
        """Should retrieve recent processing runs."""
        for i in range(5):
            memory_db.create_processing_run({'run_type': f'scan_{i}'})

        runs = memory_db.get_recent_processing_runs(limit=3)
        assert len(runs) == 3


class TestStatistics:
    """Tests for database statistics and analytics."""

    def test_get_stats_returns_counts(self, memory_db):
        """Should return file counts by type."""
        memory_db.upsert_file({'file_path': '/a.txt', 'file_hash': 'h1', 'file_type': 'text'})
        memory_db.upsert_file({'file_path': '/b.jpg', 'file_hash': 'h2', 'file_type': 'image'})
        memory_db.upsert_file({'file_path': '/c.jpg', 'file_hash': 'h3', 'file_type': 'image'})

        stats = memory_db.get_stats()
        assert stats['total_files'] == 3
        assert stats['by_type']['text'] == 1
        assert stats['by_type']['image'] == 2

    def test_get_stats_includes_categories(self, memory_db):
        """Should include category counts in stats."""
        memory_db.upsert_file({'file_path': '/a.txt', 'file_hash': 'h1', 'file_type': 'text', 'category': 'invoices'})
        memory_db.upsert_file({'file_path': '/b.txt', 'file_hash': 'h2', 'file_type': 'text', 'category': 'receipts'})
        memory_db.upsert_file({'file_path': '/c.txt', 'file_hash': 'h3', 'file_type': 'text', 'category': 'invoices'})

        stats = memory_db.get_stats()
        assert 'by_category' in stats
        assert stats['by_category']['invoices'] == 2
        assert stats['by_category']['receipts'] == 1

    def test_get_stats_includes_processing_summary(self, memory_db):
        """Should include processing statistics."""
        file_id1 = memory_db.upsert_file({'file_path': '/a.jpg', 'file_hash': 'h1', 'file_type': 'image'})
        file_id2 = memory_db.upsert_file({'file_path': '/b.jpg', 'file_hash': 'h2', 'file_type': 'image'})

        memory_db.add_detected_objects(file_id1, [{'label': 'person', 'confidence': 0.9}])
        memory_db.add_face(file_id2, {'embedding': np.random.rand(512).astype(np.float32), 'confidence': 0.95})

        stats = memory_db.get_stats()
        assert 'total_detected_objects' in stats
        assert 'total_faces' in stats


class TestTransactions:
    """Tests for transaction support."""

    def test_transaction_commit(self, memory_db):
        """Should commit transaction successfully."""
        with memory_db.transaction():
            memory_db.upsert_file({'file_path': '/a.txt', 'file_hash': 'h1', 'file_type': 'text'})
            memory_db.upsert_file({'file_path': '/b.txt', 'file_hash': 'h2', 'file_type': 'text'})

        # Both files should exist after commit
        assert memory_db.get_file('/a.txt') is not None
        assert memory_db.get_file('/b.txt') is not None

    def test_transaction_rollback(self, memory_db):
        """Should rollback transaction on error."""
        try:
            with memory_db.transaction():
                memory_db.upsert_file({'file_path': '/a.txt', 'file_hash': 'h1', 'file_type': 'text'})
                raise ValueError("Intentional error")
        except ValueError:
            pass

        # File should not exist after rollback
        assert memory_db.get_file('/a.txt') is None


class TestConcurrentAccess:
    """Tests for concurrent database access."""

    def test_multiple_connections_with_wal(self, temp_dir):
        """Should support multiple connections with WAL mode."""
        from fileforge.storage.database import Database

        db_path = temp_dir / "concurrent.db"
        db1 = Database(db_path, wal_mode=True)
        db1.initialize()

        # First connection writes and force commits
        db1.upsert_file({'file_path': '/a.txt', 'file_hash': 'h1', 'file_type': 'text'})
        db1.conn.commit()  # Explicit commit to release locks

        # Second connection should be able to read and write
        db2 = Database(db_path, wal_mode=True)
        db2.initialize()  # Sets busy_timeout
        db2.upsert_file({'file_path': '/b.txt', 'file_hash': 'h2', 'file_type': 'text'})
        db2.conn.commit()  # Explicit commit

        # Both should see all files
        assert db1.get_file('/b.txt') is not None
        assert db2.get_file('/a.txt') is not None

        # Cleanup
        db1.close()
        db2.close()


class TestDatabaseMigration:
    """Tests for database schema migrations."""

    def test_get_schema_version(self, memory_db):
        """Should track schema version."""
        version = memory_db.get_schema_version()
        assert version is not None
        assert isinstance(version, int)

    def test_upgrade_schema(self, temp_dir):
        """Should support schema upgrades."""
        from fileforge.storage.database import Database

        db = Database(temp_dir / "test.db")
        db.initialize()

        initial_version = db.get_schema_version()

        # Migration should be idempotent
        db.migrate_schema()

        final_version = db.get_schema_version()
        assert final_version >= initial_version
