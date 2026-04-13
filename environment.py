import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Patient:
    patient_id: int
    position: Tuple[int, int]
    urgency: int
    deadline: int
    assigned: bool = False
    served: bool = False
    expired: bool = False


@dataclass
class Ambulance:
    ambulance_id: int
    position: Tuple[int, int]
    idle: bool = True
    busy_timer: int = 0
    assigned_patient_id: Optional[int] = None


class HospitalDispatchEnv:
    """
    GridWorld-style hospital dispatch environment.
    Dispatch decisions are allocation decisions, not movement navigation.
    Travel time is computed with Manhattan distance.
    """

    def __init__(
        self,
        grid_size: int = 5,
        num_patients: int = 5,
        num_ambulances: int = 2,
        max_steps: int = 25,
        hospital_position: Tuple[int, int] = (2, 2),
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.num_patients = num_patients
        self.num_ambulances = num_ambulances
        self.max_steps = max_steps
        self.hospital_position = hospital_position
        self.random = random.Random(seed)

        self.current_step = 0
        self.patients: Dict[int, Patient] = {}
        self.ambulances: Dict[int, Ambulance] = {}

        self.total_response_time = 0
        self.completed_patients = 0
        self.expired_patients = 0
        self.total_busy_time = 0

        self.reset()

    def reset(self) -> Tuple:
        self.current_step = 0
        self.total_response_time = 0
        self.completed_patients = 0
        self.expired_patients = 0
        self.total_busy_time = 0

        self.patients = {}
        self.ambulances = {}

        # Initialize ambulances at the hospital
        for amb_id in range(self.num_ambulances):
            self.ambulances[amb_id] = Ambulance(
                ambulance_id=amb_id,
                position=self.hospital_position,
                idle=True,
                busy_timer=0,
                assigned_patient_id=None,
            )

        # Place patients randomly, avoiding hospital position duplicates if possible
        used_positions = {self.hospital_position}
        for patient_id in range(self.num_patients):
            while True:
                pos = (
                    self.random.randint(0, self.grid_size - 1),
                    self.random.randint(0, self.grid_size - 1),
                )
                if pos not in used_positions:
                    used_positions.add(pos)
                    break

            urgency = self.random.choice([1, 2])  # 2 = high urgency
            deadline = 5 if urgency == 2 else 8

            self.patients[patient_id] = Patient(
                patient_id=patient_id,
                position=pos,
                urgency=urgency,
                deadline=deadline,
            )

        return self.get_state()

    def manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_active_patients(self) -> List[Patient]:
        return [
            p
            for p in self.patients.values()
            if not p.served and not p.expired
        ]

    def get_idle_ambulances(self) -> List[Ambulance]:
        return [a for a in self.ambulances.values() if a.idle]

    def get_state(self) -> Tuple:
        active_patients = self.get_active_patients()
        idle_ambulances = self.get_idle_ambulances()

        num_active = len(active_patients)

        nearest_patient_distance = (
            min(
                self.manhattan_distance(self.hospital_position, p.position)
                for p in active_patients
            )
            if active_patients
            else 0
        )

        max_urgency = max((p.urgency for p in active_patients), default=0)
        min_deadline = min((p.deadline for p in active_patients), default=0)

        # Deadline bucket compresses state space
        if min_deadline <= 2:
            deadline_bucket = 0
        elif min_deadline <= 5:
            deadline_bucket = 1
        else:
            deadline_bucket = 2

        state = (
            int(self.ambulances[0].idle),
            int(self.ambulances[1].idle),
            self.ambulances[0].position[0],
            self.ambulances[0].position[1],
            self.ambulances[1].position[0],
            self.ambulances[1].position[1],
            num_active,
            nearest_patient_distance,
            max_urgency,
            deadline_bucket,
            len(idle_ambulances),
        )
        return state

    def get_valid_actions(self) -> List[Tuple[str, Optional[int], Optional[int]]]:
        """
        Action format:
        ('dispatch', ambulance_id, patient_id)
        ('hold', None, None)
        """
        actions = [('hold', None, None)]

        active_patients = self.get_active_patients()
        idle_ambulances = self.get_idle_ambulances()

        for amb in idle_ambulances:
            for patient in active_patients:
                if not patient.assigned:
                    actions.append(('dispatch', amb.ambulance_id, patient.patient_id))

        return actions

    def step(self, action: Tuple[str, Optional[int], Optional[int]]) -> Tuple[Tuple, float, bool, Dict]:
        reward = 0.0
        info = {}

        action_type, ambulance_id, patient_id = action

        # 1. Advance existing ambulance tasks
        for amb in self.ambulances.values():
            if not amb.idle:
                amb.busy_timer -= 1
                self.total_busy_time += 1

                if amb.busy_timer <= 0:
                    served_patient_id = amb.assigned_patient_id
                    if served_patient_id is not None:
                        patient = self.patients[served_patient_id]
                        patient.served = True
                        patient.assigned = False
                        self.completed_patients += 1
                        reward += 20  # delivered to hospital

                    amb.idle = True
                    amb.assigned_patient_id = None
                    amb.position = self.hospital_position

        # 2. Reduce deadlines for active unassigned/unserved patients
        for patient in self.get_active_patients():
            if not patient.assigned:
                patient.deadline -= 1
                if patient.deadline <= 0:
                    patient.expired = True
                    self.expired_patients += 1
                    reward -= 15

        # 3. Execute current dispatch decision
        active_patients = self.get_active_patients()

        if action_type == 'dispatch':
            if ambulance_id is None or patient_id is None:
                reward -= 5
            else:
                ambulance = self.ambulances[ambulance_id]
                patient = self.patients[patient_id]

                if not ambulance.idle:
                    reward -= 5
                elif patient.served or patient.expired or patient.assigned:
                    reward -= 5
                else:
                    to_patient = self.manhattan_distance(ambulance.position, patient.position)
                    to_hospital = self.manhattan_distance(patient.position, self.hospital_position)
                    total_trip_time = to_patient + to_hospital

                    ambulance.idle = False
                    ambulance.assigned_patient_id = patient_id
                    ambulance.busy_timer = max(1, total_trip_time)

                    patient.assigned = True

                    self.total_response_time += to_patient
                    reward += 10  # pickup success
                    reward -= to_patient  # penalize slower response
                    reward += 3 * patient.urgency  # urgency bonus

        elif action_type == 'hold':
            urgent_unassigned_exists = any(
                p.urgency == 2 and not p.assigned
                for p in active_patients
            )
            if urgent_unassigned_exists and len(self.get_idle_ambulances()) > 0:
                reward -= 3

        self.current_step += 1

        done = (
            self.current_step >= self.max_steps
            or all(p.served or p.expired for p in self.patients.values())
        )

        next_state = self.get_state()

        info["completed_patients"] = self.completed_patients
        info["expired_patients"] = self.expired_patients

        return next_state, reward, done, info

    def render(self) -> None:
        print(f"\nStep: {self.current_step}")
        print(f"Hospital: {self.hospital_position}")
        print("Ambulances:")
        for amb in self.ambulances.values():
            print(
                f"  Ambulance {amb.ambulance_id}: pos={amb.position}, "
                f"idle={amb.idle}, busy_timer={amb.busy_timer}, assigned_patient={amb.assigned_patient_id}"
            )

        print("Patients:")
        for patient in self.patients.values():
            print(
                f"  Patient {patient.patient_id}: pos={patient.position}, urgency={patient.urgency}, "
                f"deadline={patient.deadline}, assigned={patient.assigned}, "
                f"served={patient.served}, expired={patient.expired}"
            )