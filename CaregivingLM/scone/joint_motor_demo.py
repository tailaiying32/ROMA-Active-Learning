import numpy as np
import time
from sconetools import sconepy

def run_joint_motor_demo():
    # Set the SCONE log level
    sconepy.set_log_level(3)
    print("SCONE Version", sconepy.version())

    # Load the joint actuator model
    model_file = "models/HSA13T_hfd_joint.scone"
    print(f"Loading model: {model_file}")
    model = sconepy.load_model(model_file)

    # Print model info
    print(f"Model loaded: {model.name()}")
    print(f"Number of DOFs: {len(model.dofs())}")
    print(f"Number of joint motors: {len(model.actuators())}")

    # List all DOFs and actuators
    print("\nDOFs:")
    for dof in model.dofs():
        print(f"  {dof.name()}: pos={dof.pos():.3f}")

    print("\nActuators (Joint Motors):")
    for i, act in enumerate(model.actuators()):
        print(f"  {i}: {act.name()}")

    # Reset and initialize the model
    model.reset()
    model.set_store_data(True)

    # # Set initial pose - arm in neutral position
    # for dof in model.dofs():
    #     if dof.name() == "elbow_flexion_r":
    #         dof.set_pos(np.radians(90))  # 90 degrees elbow flexion
    #     elif dof.name() == "shoulder_abduction_r":
    #         dof.set_pos(np.radians(45))  # 45 degrees shoulder abduction
    #     elif dof.name() == "shoulder_flexion_r":
    #         dof.set_pos(np.radians(30))  # 30 degrees shoulder flexion
    #     else:
    #         dof.set_pos(0.0)  # All other joints at neutral

    # Apply the DOF positions
    model.init_state_from_dofs()

    print(f"\nStarting simulation...")

    # Phase 1: Let the model settle for 1 second with zero joint motor inputs
    print("Phase 1: Settling for 1.0 second...")
    settle_time = 1.0
    dt = 0.01

    for t in np.arange(0, settle_time, dt):
        # Set all joint motor inputs to zero during settling
        zero_inputs = np.zeros(len(model.actuators()))
        model.set_actuator_inputs(zero_inputs)

        # Advance simulation
        model.advance_simulation_to(t)

    print(f"Settling complete at t={model.time():.3f}s")

    # Find the elbow joint motor
    elbow_motor_index = None
    for i, act in enumerate(model.actuators()):
        if "elbow" in act.name().lower():
            elbow_motor_index = i
            print(f"Found elbow motor at index {i}: {act.name()}")
            break

    if elbow_motor_index is None:
        print("Could not find elbow joint motor!")
        return

    # Phase 2: Apply input to elbow joint motor and simulate
    print("Phase 2: Applying input to elbow joint motor...")
    simulation_time = 2.0  # Additional simulation time
    elbow_input = 3  # Normalized input value for joint motor

    for t in np.arange(settle_time, settle_time + simulation_time, dt):
        # Set joint motor inputs
        motor_inputs = np.zeros(len(model.actuators()))
        motor_inputs[elbow_motor_index] = elbow_input

        model.set_actuator_inputs(motor_inputs)

        # Advance simulation
        model.advance_simulation_to(t)

        # Print progress every 0.5 seconds
        if t % 0.5 < dt:
            elbow_dof = None
            for dof in model.dofs():
                if dof.name() == "elbow_flexion_r":
                    elbow_dof = dof
                    break
            if elbow_dof:
                print(f"  t={t:.3f}s: Elbow angle = {np.degrees(elbow_dof.pos()):.1f}°")

    print(f"Simulation complete at t={model.time():.3f}s")

    # Save results
    dirname = "joint_motor_demo"
    filename = f"elbow_input_{elbow_input}_{model.time():.3f}s"
    model.write_results(dirname, filename)
    print(f"Results saved to {dirname}/{filename}.sto")
    print(f"Open the .sto file in SCONE Studio to visualize the results")

    # Print final joint angles
    print("\nFinal joint angles:")
    for dof in model.dofs():
        print(f"  {dof.name()}: {np.degrees(dof.pos()):.1f}°")

if __name__ == "__main__":
    run_joint_motor_demo()