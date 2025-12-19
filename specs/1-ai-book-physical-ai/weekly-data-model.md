# Data Model: Weekly Development Plan - AI-Powered Book

## Overview
This document defines the data models for the project management aspects of the 13-week development plan, including entities related to scheduling, milestones, and progress tracking.

## Project Management Entities

### Week
**Description**: Represents a single week in the 13-week development plan

**Fields**:
- `id` (Integer, Primary Key): Week number (1-13)
- `start_date` (Date, Required): Start date of the week
- `end_date` (Date, Required): End date of the week
- `title` (String, Required): Week title/description
- `status` (String, Default: "planned"): Week status ("planned", "in_progress", "completed", "delayed")
- `created_at` (DateTime, Required): Record creation timestamp
- `updated_at` (DateTime, Required): Last update timestamp

**Validation Rules**:
- `id` must be between 1 and 13
- `start_date` must be before `end_date`
- `status` must be one of the allowed values

### Task
**Description**: Individual tasks within each week of the development plan

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the task
- `week_id` (Integer, Foreign Key): Reference to the Week
- `title` (String, Required): Task title
- `description` (Text, Optional): Detailed task description
- `priority` (String, Default: "medium"): Task priority ("low", "medium", "high", "critical")
- `status` (String, Default: "pending"): Task status ("pending", "in_progress", "completed", "blocked")
- `estimated_hours` (Integer, Required): Estimated time to complete in hours
- `actual_hours` (Integer, Optional): Actual time spent in hours
- `assignee` (String, Optional): Team member assigned to the task
- `dependencies` (JSON, Optional): List of task IDs this task depends on
- `created_at` (DateTime, Required): Task creation timestamp
- `updated_at` (DateTime, Required): Last update timestamp

**Validation Rules**:
- `week_id` must reference a valid Week
- `priority` must be one of the allowed values
- `status` must be one of the allowed values
- `estimated_hours` must be positive

### Milestone
**Description**: Major checkpoints and deliverables in the development plan

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the milestone
- `week_id` (Integer, Foreign Key): Reference to the Week (target completion week)
- `title` (String, Required): Milestone title
- `description` (Text, Required): Detailed milestone description
- `type` (String, Required): Milestone type ("feature", "integration", "testing", "deployment")
- `status` (String, Default: "pending"): Milestone status ("pending", "in_progress", "achieved", "delayed")
- `deliverable` (String, Required): Specific deliverable expected
- `success_criteria` (JSON, Required): Criteria for milestone success
- `created_at` (DateTime, Required): Milestone creation timestamp
- `updated_at` (DateTime, Required): Last update timestamp

**Validation Rules**:
- `week_id` must reference a valid Week
- `type` must be one of the allowed values
- `status` must be one of the allowed values
- `success_criteria` must be valid JSON

### Dependency
**Description**: Relationships between tasks and weeks showing dependencies

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the dependency
- `from_task_id` (UUID, Foreign Key): Task that must be completed first
- `to_task_id` (UUID, Foreign Key): Task that depends on the first task
- `dependency_type` (String, Default: "finish_to_start"): Type of dependency ("finish_to_start", "start_to_start", "finish_to_finish", "start_to_finish")
- `created_at` (DateTime, Required): Dependency creation timestamp

**Validation Rules**:
- `from_task_id` and `to_task_id` must reference valid Tasks
- `dependency_type` must be one of the allowed values
- A task cannot depend on itself

### ProgressReport
**Description**: Weekly progress reports and status updates

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the report
- `week_id` (Integer, Foreign Key): Reference to the Week
- `report_date` (DateTime, Required): Date the report was created
- `completed_tasks` (JSON, Required): List of completed task IDs
- `in_progress_tasks` (JSON, Required): List of in-progress task IDs
- `blocked_tasks` (JSON, Required): List of blocked task IDs
- `issues_identified` (JSON, Optional): List of issues discovered
- `risks_assessed` (JSON, Optional): List of risks identified
- `next_week_focus` (Text, Optional): Focus areas for next week
- `overall_status` (String, Required): Overall week status ("on_track", "delayed", "ahead", "at_risk")
- `created_at` (DateTime, Required): Report creation timestamp

**Validation Rules**:
- `week_id` must reference a valid Week
- `overall_status` must be one of the allowed values
- All task IDs in JSON fields must reference valid Tasks

### Feature
**Description**: Major features being developed during the 13-week plan

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the feature
- `name` (String, Required): Feature name
- `description` (Text, Required): Feature description
- `type` (String, Required): Feature type ("base", "bonus")
- `complexity` (String, Required): Complexity level ("low", "medium", "high")
- `start_week` (Integer, Required): Week when development starts
- `target_week` (Integer, Required): Target week for completion
- `status` (String, Default: "planned"): Feature status ("planned", "in_development", "testing", "completed")
- `dependencies` (JSON, Optional): List of feature IDs this feature depends on
- `created_at` (DateTime, Required): Feature creation timestamp
- `updated_at` (DateTime, Required): Last update timestamp

**Validation Rules**:
- `type` must be either "base" or "bonus"
- `complexity` must be one of the allowed values
- `start_week` and `target_week` must be between 1 and 13
- `status` must be one of the allowed values

### TestPlan
**Description**: Testing activities planned for each week

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the test plan
- `week_id` (Integer, Foreign Key): Reference to the Week
- `feature_id` (UUID, Foreign Key): Reference to the Feature being tested
- `test_type` (String, Required): Type of testing ("unit", "integration", "end_to_end", "performance", "security")
- `test_scenarios` (JSON, Required): List of test scenarios to execute
- `responsible_team` (String, Required): Team responsible for testing
- `status` (String, Default: "planned"): Test status ("planned", "in_progress", "completed", "passed", "failed")
- `created_at` (DateTime, Required): Test plan creation timestamp
- `updated_at` (DateTime, Required): Last update timestamp

**Validation Rules**:
- `week_id` must reference a valid Week
- `feature_id` must reference a valid Feature
- `test_type` must be one of the allowed values
- `status` must be one of the allowed values

## Relationships

### Week → Task (One-to-Many)
- One Week can have many Tasks
- Tasks are deleted when Week is deleted (CASCADE)

### Week → Milestone (One-to-Many)
- One Week can have many Milestones
- Milestones remain when Week is deleted (SET NULL)

### Week → ProgressReport (One-to-One)
- One Week has one ProgressReport
- ProgressReport is deleted when Week is deleted (CASCADE)

### Task → Dependency (Many-to-Many via Dependency table)
- Tasks can have multiple dependencies and be depended on by multiple tasks

### Feature → Task (One-to-Many)
- One Feature can have many Tasks
- Tasks remain when Feature is deleted (SET NULL)

### Feature → TestPlan (One-to-Many)
- One Feature can have many TestPlans
- TestPlans are deleted when Feature is deleted (CASCADE)

### Week → TestPlan (One-to-Many)
- One Week can have many TestPlans
- TestPlans remain when Week is deleted (SET NULL)

## Indexes

### Week
- Index on `id` (unique) for week identification
- Index on `start_date` for chronological queries

### Task
- Index on `week_id` for week-specific queries
- Index on `status` for status-based filtering
- Index on `priority` for priority-based queries

### Milestone
- Index on `week_id` for milestone scheduling
- Index on `status` for milestone tracking
- Index on `type` for milestone categorization

### Dependency
- Index on `from_task_id` and `to_task_id` for dependency resolution
- Index on `dependency_type` for dependency type queries

### ProgressReport
- Index on `week_id` (unique) for weekly reports
- Index on `report_date` for chronological queries

### Feature
- Index on `type` for feature categorization
- Index on `status` for feature tracking
- Index on `target_week` for deadline tracking

### TestPlan
- Index on `week_id` and `feature_id` for test scheduling
- Index on `test_type` for test categorization
- Index on `status` for test tracking

## Constraints

### Referential Integrity
- Foreign key constraints enforce relationship validity
- CASCADE and SET NULL behaviors are explicitly defined

### Data Integrity
- NOT NULL constraints on required fields
- CHECK constraints where appropriate for value validation
- UNIQUE constraints to prevent duplicates where required

### Business Logic
- Week IDs must be sequential from 1 to 13
- Task dependencies must not create circular references
- Milestone target weeks must align with feature development timeline
- Test plans must align with feature completion schedule