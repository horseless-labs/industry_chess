# Industry Chess

Industry Chess is a computer vision testbed for real-time process validation in a controlled physical environment. It uses a chessboard as a structured sandbox to demonstrate how visual input can be converted into validated state transitions and structured process data.

Chess is not the objective, just the is the constrained environment used to prototype validation systems. This project explores how computer vision can be used to:

* Validate whether a task was completed correctly
* Detect missing, misplaced, or incorrect components
* Track process state transitions automatically
* Measure cycle time per operation
* Generate structured data for downstream analytics

The chessboard provides a deterministic, rule-constrained system that allows reliable validation logic to be developed before deployment in industrial settings.

## Overview

The system observes a physical chessboard through a camera feed and performs:

* Board detection and geometric normalization
* Piece detection and classification
* Coordinate mapping
* Structured state generation (FEN)
* State comparison and move inference
* Timestamped event logging

The result is a continuously updated, machine-readable representation of physical state.

## System Pipeline

Camera Input
→ Board Detection (Homography)
→ Piece Detection (YOLO)
→ Square Mapping
→ FEN State Generation
→ State Comparison
→ Move Validation
→ Event Logging

Each frame is converted into structured state. Each state transition is validated and recorded.

## Industrial Analogy

In a manufacturing context, the same architecture can be applied to:

* Assembly verification
* Fixture state monitoring
* Step-by-step procedural enforcement
* Quality-control validation
* Cycle-time analysis

Instead of relying on manual logging or operator confirmation, state is inferred directly from observed physical configuration.

## Example Outputs

* FEN string representing full board state
* Detected move (e.g., e2 → e4)
* Timestamped state transitions
* Piece count validation
* Cycle-time measurement between moves

All outputs are structured and machine-readable.

## Tech Stack

* Python
* OpenCV
* YOLO
* NumPy
* python-chess

---

## Current Status

* Real-time board detection is spotty; proceeded through demo with manual corner selection
* Piece detection functional
* FEN generation stable
* Basic event logging active; an accurate board and score can be generated, but database or analysis logic is yet to be implemented

---

## Roadmap

* Improve detection robustness under lighting variation
* Add anomaly detection layer
* Expand structured logging and export formats
* Fix corner detection
* Integrate hardware manipulation (gantry / robotic arm)
* Implement closed-loop validation

---

## Design Principle

Physical state → Structured representation → Validation → Logged event

The core problem being solved is reliable extraction of enforceable state from visual input.

Industry Chess exists to test and refine that capability.
