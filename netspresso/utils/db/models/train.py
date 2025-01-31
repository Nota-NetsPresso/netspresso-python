from sqlalchemy import JSON, Boolean, Column, Float, ForeignKey, Integer, String, BigInteger
from sqlalchemy.orm import relationship

from netspresso.utils.db.generate_uuid import generate_uuid
from netspresso.utils.db.mixins import TimestampMixin
from netspresso.utils.db.session import Base


class Augmentation(Base):
    __tablename__ = "augmentation"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    name = Column(String(50), nullable=False)
    parameters = Column(JSON, nullable=False)
    phase = Column(String(30), nullable=False) # train, inference

    hyperparameter_id = Column(Integer, ForeignKey("hyperparameter.id"), nullable=False)
    hyperparameter = relationship("Hyperparameter", back_populates="augmentations", lazy='joined')


class TrainTask(Base, TimestampMixin):
    __tablename__ = "train_task"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    task_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="task"))
    pretrained_model = Column(String(100), nullable=False)
    task = Column(String(30), nullable=False)
    framework = Column(String(30), nullable=False)
    input_shapes = Column(JSON, nullable=False)
    status = Column(String(30), nullable=False)
    error_detail = Column(JSON, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)

    # Relationships (1:1 Mapping)
    dataset = relationship("Dataset", back_populates="task", uselist=False, cascade="all, delete-orphan", lazy='joined')
    hyperparameter = relationship("Hyperparameter", back_populates="task", uselist=False, cascade="all, delete-orphan", lazy='joined')
    environment = relationship("Environment", back_populates="task", uselist=False, cascade="all, delete-orphan", lazy='joined')
    performance = relationship("Performance", back_populates="task", uselist=False, cascade="all, delete-orphan", lazy='joined')

    # Relationship to TrainedModel
    model = relationship(
        "TrainedModel",
        back_populates="train_task",
        uselist=False,
        lazy='joined',
    )


class Dataset(Base, TimestampMixin):
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    train_path = Column(String(255), nullable=False)
    valid_path = Column(String(255), nullable=True)
    test_path = Column(String(255), nullable=True)
    storage_location = Column(String(50), nullable=False)
    id_mapping = Column(JSON, nullable=True)
    palette = Column(JSON, nullable=True)

    # Relationship to TrainTask
    task_id = Column(String(36), ForeignKey("train_task.task_id", ondelete="CASCADE"), unique=True, nullable=False)
    task = relationship("TrainTask", back_populates="dataset")


class Hyperparameter(Base, TimestampMixin):
    __tablename__ = "hyperparameter"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    epochs = Column(Integer, nullable=False, default=0)
    batch_size = Column(Integer, nullable=False)
    optimizer = Column(JSON, nullable=True)
    scheduler = Column(JSON, nullable=True)

    augmentations = relationship("Augmentation", back_populates="hyperparameter", cascade="all, delete-orphan", lazy='joined')

    # Relationship to TrainTask
    task_id = Column(String(36), ForeignKey("train_task.task_id", ondelete="CASCADE"), unique=True, nullable=False)
    task = relationship("TrainTask", back_populates="hyperparameter")


class Environment(Base, TimestampMixin):
    __tablename__ = "environment"

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    seed = Column(Integer, nullable=False)
    num_workers = Column(Integer, nullable=False)
    gpus = Column(String(30), nullable=False)  # GPUs (예: "1, 0")

    # Relationship to TrainTask
    task_id = Column(String(36), ForeignKey("train_task.task_id", ondelete="CASCADE"), unique=True, nullable=False)
    task = relationship("TrainTask", back_populates="environment")


class Performance(Base, TimestampMixin):
    __tablename__ = "performance"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    train_losses = Column(JSON, nullable=False)
    valid_losses = Column(JSON, nullable=False)
    train_metrics = Column(JSON, nullable=False)
    valid_metrics = Column(JSON, nullable=False)
    metrics_list = Column(JSON, nullable=False)
    primary_metric = Column(String(36), nullable=False)
    flops = Column(String(50), nullable=False, default=0)
    params = Column(String(50), nullable=False, default=0)
    total_train_time = Column(Float, nullable=False, default=0)
    best_epoch = Column(Integer, nullable=False, default=0)
    last_epoch = Column(Integer, nullable=False, default=0)
    total_epoch = Column(Integer, nullable=False, default=0)
    status = Column(String(36), nullable=True)

    # Relationship to TrainTask
    task_id = Column(String(36), ForeignKey("train_task.task_id", ondelete="CASCADE"), unique=True, nullable=False)
    task = relationship("TrainTask", back_populates="performance")
