import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

CHALLENGES = {
    1: "Level 1: Prepare |00⟩ (both qubits 0)",
    2: "Level 2: Prepare |11⟩ (both qubits 1)",
    3: "Level 3: Prepare |++⟩ (both qubits in superposition)",
    4: "Level 4: Create Bell state |Φ⁺⟩",
    5: "Level 5: Create phase-entangled state (|01⟩ + i|10⟩)/√2"
}

class QuantumGame2Q:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum State Builder")
        self.root.geometry("1400x900")

        self.level = 1
        self.score = 0
        self.gates_used = 0
        self.history = []
        
        self.backend = AerSimulator()
        self.circuit = None
        self.statevector = None

        # UI state
        self.active_qubit = tk.IntVar(value=0)
        self.cx_control = tk.IntVar(value=0)
        self.cx_target = tk.IntVar(value=1)

        self.build_ui()
        self.new_circuit()

    # ==============================================================
    # UI BUILDING
    # ==============================================================

    def build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Quantum State Builder", font=('Arial',16,'bold')).pack()
        info = ttk.Frame(top)
        info.pack()

        self.level_lbl = ttk.Label(info, text=f"Level: {self.level}")
        self.level_lbl.pack(side=tk.LEFT, padx=10)

        self.score_lbl = ttk.Label(info, text=f"Score: {self.score}")
        self.score_lbl.pack(side=tk.LEFT, padx=10)

        self.gate_lbl = ttk.Label(info, text=f"Gates Used: {self.gates_used}")
        self.gate_lbl.pack(side=tk.LEFT, padx=10)

        btns = ttk.Frame(top)
        btns.pack(pady=6)

        for text, cmd in [
            ("New Circuit", self.new_circuit),
            ("Measure", self.measure),
            ("Check State", self.check_state),
            ("Next Level", self.next_level),
            ("Reset Game", self.reset_game),
            ("Make Bell", self.make_bell)
        ]:
            ttk.Button(btns, text=text, command=cmd).pack(side=tk.LEFT, padx=6)

        # Main layout
        main = ttk.Frame(self.root)
        main.pack(expand=True, fill=tk.BOTH)

        # Left: plots
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.fig = plt.Figure(figsize=(9,6), dpi=100)
        self.ax0 = self.fig.add_subplot(2,2,1, projection='3d')
        self.ax1 = self.fig.add_subplot(2,2,2, projection='3d')
        self.ax_prob = self.fig.add_subplot(2,1,2)

        self.canvas = FigureCanvasTkAgg(self.fig, left)
        self.canvas.get_tk_widget().pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # Right column
        right = ttk.Frame(main, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(right, text="Quantum Gates", font=('Arial',12,'bold')).pack(pady=5)
        self.make_gate_panel(right)

        ttk.Label(right, text="Active Qubit", font=('Arial',10,'bold')).pack(pady=(10,0))
        q_frame = ttk.Frame(right); q_frame.pack()
        ttk.Radiobutton(q_frame, text="Qubit 0", variable=self.active_qubit, value=0).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(q_frame, text="Qubit 1", variable=self.active_qubit, value=1).pack(side=tk.LEFT, padx=5)

        ttk.Label(right, text="CNOT control/target", font=('Arial',10,'bold')).pack(pady=(10,0))
        cx_frame = ttk.Frame(right); cx_frame.pack()
        ttk.Label(cx_frame, text="Control").grid(row=0,column=0,padx=4)
        ttk.Combobox(cx_frame, textvariable=self.cx_control, values=[0,1], width=3).grid(row=0,column=1)
        ttk.Label(cx_frame, text="Target").grid(row=0,column=2,padx=4)
        ttk.Combobox(cx_frame, textvariable=self.cx_target, values=[0,1], width=3).grid(row=0,column=3)

        # Text areas
        self.challenge_box = tk.Text(right, width=40, height=5, font=('Arial',10))
        self.challenge_box.pack(pady=10, fill=tk.X)

        ttk.Label(right, text="Game History", font=('Arial',10,'bold')).pack()
        self.hist_box = tk.Text(right, width=40, height=8, font=('Arial',9))
        self.hist_box.pack(pady=5, fill=tk.X)

        ttk.Label(right, text="Current State Info", font=('Arial',10,'bold')).pack()
        self.state_box = tk.Text(right, width=40, height=14, font=('Arial',9))
        self.state_box.pack(pady=5, fill=tk.X)

        self.update_challenge()

    def make_gate_panel(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(pady=5)

        gates = [
            ("X", "Bit flip"),
            ("Y", "Bit + phase flip"),
            ("Z", "Phase flip"),
            ("H", "Superposition"),
            ("S", "Phase π/2"),
            ("T", "Phase π/4"),
            ("CX", "Controlled NOT")
        ]

        for name, desc in gates:
            row = ttk.Frame(frame)
            row.pack(pady=3, fill=tk.X)
            ttk.Button(row, text=name, width=5,
                       command=lambda n=name: self.apply_gate(n)).pack(side=tk.LEFT)
            ttk.Label(row, text=name).pack(side=tk.LEFT, padx=6)
            ttk.Label(row, text=desc, foreground='gray', font=('Arial',7)).pack(side=tk.LEFT, padx=6)

    # ==============================================================
    # CIRCUIT LOGIC
    # ==============================================================

    def new_circuit(self):
        self.circuit = QuantumCircuit(2, 2)
        self.statevector = Statevector.from_instruction(self.circuit)
        self.gates_used = 0
        self.history.clear()
        self.add_history("New 2-qubit circuit initialized (|00⟩)")
        self.update_ui()

    def apply_gate(self, gate):
        aq = int(self.active_qubit.get())

        if gate == "CX":
            ctrl = int(self.cx_control.get())
            tgt = int(self.cx_target.get())
            if ctrl == tgt:
                messagebox.showwarning("Error", "Control and target must be different")
                return
            self.circuit.cx(ctrl, tgt)
            self.add_history(f"Applied CX(control={ctrl}, target={tgt})")
        else:
            getattr(self.circuit, gate.lower())(aq)
            self.add_history(f"Applied {gate} on qubit {aq}")

        self.gates_used += 1
        self.statevector = Statevector.from_instruction(self.circuit)
        self.update_ui()

    def make_bell(self):
        self.circuit.h(0)
        self.circuit.cx(0,1)
        self.statevector = Statevector.from_instruction(self.circuit)
        self.add_history("Created Bell state |Φ⁺⟩")
        self.gates_used += 2
        self.update_ui()

    def measure(self):
        m = self.circuit.copy()
        m.measure([0,1], [0,1])
        job = self.backend.run(transpile(m, self.backend), shots=200)
        counts = job.result().get_counts()
        self.add_history(f"Measurement (200 shots): {counts}")
        self.update_ui()

    # ==============================================================
    # SUCCESS CHECKING
    # ==============================================================

    def check_state(self):
        success, fidelity = self.check_success()
        if success:
            pts = int(fidelity * 100)
            self.score += pts
            self.add_history(f"SUCCESS! Fidelity={fidelity:.3f}, +{pts} points")
            messagebox.showinfo("Success!", f"Fidelity={fidelity:.3f}\n+{pts} points!")
        else:
            self.add_history(f"Not yet... Fidelity={fidelity:.3f}")
            messagebox.showinfo("Try again", f"Fidelity={fidelity:.3f}")

    def check_success(self):
        sv = self.statevector.data

        if self.level == 1:
            target = np.array([1,0,0,0], dtype=complex)
        elif self.level == 2:
            target = np.array([0,0,0,1], dtype=complex)
        elif self.level == 3:
            target = (1/2)*np.array([1,1,1,1], dtype=complex)
        elif self.level == 4:
            target = (1/np.sqrt(2))*np.array([1,0,0,1], dtype=complex)
        else:
            target = (1/np.sqrt(2))*np.array([0,1,1j,0], dtype=complex)

        target /= np.linalg.norm(target)
        sv /= np.linalg.norm(sv)

        fidelity = abs(np.vdot(target, sv))**2
        return (fidelity > 0.85), fidelity

    def next_level(self):
        ok, f = self.check_success()
        if ok:
            if self.level < 5:
                self.level += 1
                self.add_history(f"Advanced to Level {self.level}")
                self.new_circuit()
                self.update_challenge()
                messagebox.showinfo("Level Up!", CHALLENGES[self.level])
            else:
                messagebox.showinfo("Completed!", "You beat all levels!")
        else:
            messagebox.showinfo("Not ready", f"Fidelity {f:.3f} < 0.85")

    def reset_game(self):
        self.level = 1
        self.score = 0
        self.gates_used = 0
        self.history.clear()
        self.new_circuit()
        self.update_challenge()

    # ==============================================================
    # BLOCH SPHERE + GLOW EFFECT
    # ==============================================================

    def reduced_density(self, psi, keep):
        rho = np.outer(psi, np.conj(psi))
        out = np.zeros((2,2), dtype=complex)

        if keep == 0:  # keep q0, trace over q1
            for q1 in [0,1]:
                for a in [0,1]:
                    for a2 in [0,1]:
                        i = q1*2 + a
                        j = q1*2 + a2
                        out[a,a2] += rho[i,j]

        else:  # keep q1, trace over q0
            for q0 in [0,1]:
                for a in [0,1]:
                    for a2 in [0,1]:
                        i = a*2 + q0
                        j = a2*2 + q0
                        out[a,a2] += rho[i,j]

        return out

    def bloch_vector(self, rho):
        X = np.array([[0,1],[1,0]], complex)
        Y = np.array([[0,-1j],[1j,0]], complex)
        Z = np.array([[1,0],[0,-1]], complex)
        return np.array([
            np.real(np.trace(rho @ X)),
            np.real(np.trace(rho @ Y)),
            np.real(np.trace(rho @ Z))
        ])

    # ----------- GLOW EFFECT HERE ------------------

    def _draw_glow(self, ax, intensity):
        # Cyan glowing sphere at center
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = 0.25*np.outer(np.cos(u), np.sin(v))
        y = 0.25*np.outer(np.sin(u), np.sin(v))
        z = 0.25*np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(
            x, y, z,
            color=(0,1,1,intensity),
            linewidth=0,
            shade=True,
            alpha=0.25 * intensity
        )

        ax.scatter([0],[0],[0], color='cyan', s=300*intensity, alpha=0.6*intensity)

        ax.text2D(
            0.05, 0.9,
            "Entangled / Mixed",
            transform=ax.transAxes,
            color=(0,1,1,intensity),
            fontsize=10,
            fontweight='bold'
        )

    def _draw_bloch(self, ax, vec, title):
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))

        ax.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.5, alpha=0.6)

        length = np.linalg.norm(vec)

        # Glow intensity increases as vector shrinks
        glow = max(0, (0.35 - length) / 0.35)
        if glow > 0:
            self._draw_glow(ax, glow)

        ax.quiver(
            0,0,0, vec[0], vec[1], vec[2],
            length=1,
            linewidth=3,
            color='blue',
            arrow_length_ratio=0.08
        )

        ax.set_title(f"{title} (|r|={length:.2f})")
        ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # ==============================================================
    # PLOTTING
    # ==============================================================

    def update_plots(self):
        self.ax0.cla()
        self.ax1.cla()
        self.ax_prob.cla()

        sv = self.statevector.data
        probs = np.abs(sv)**2

        rho0 = self.reduced_density(sv, keep=0)
        rho1 = self.reduced_density(sv, keep=1)

        v0 = self.bloch_vector(rho0)
        v1 = self.bloch_vector(rho1)

        self._draw_bloch(self.ax0, v0, "Qubit 0")
        self._draw_bloch(self.ax1, v1, "Qubit 1")

        labels = ['00','01','10','11']
        self.ax_prob.bar(labels, probs)
        self.ax_prob.set_ylim([0,1])
        self.ax_prob.set_ylabel("Probability")
        self.ax_prob.set_title("Joint State Probabilities")

        self.fig.tight_layout()
        self.canvas.draw()

    # ==============================================================
    # STATE TEXT & HISTORY
    # ==============================================================

    def update_state_text(self):
        sv = self.statevector.data
        probs = np.abs(sv)**2
        fidelity = self.check_success()[1]

        text = f"Level {self.level} – Fidelity: {fidelity:.3f}\n\n"
        text += "Amplitudes (|q1 q0>):\n"
        for i, amp in enumerate(sv):
            text += f"  {i}: {amp:.3f}  prob={probs[i]:.2%}\n"

        text += "\nProbabilities:\n"
        text += f"  |00>: {probs[0]:.2%}\n"
        text += f"  |01>: {probs[1]:.2%}\n"
        text += f"  |10>: {probs[2]:.2%}\n"
        text += f"  |11>: {probs[3]:.2%}\n"

        self.state_box.delete(1.0, tk.END)
        self.state_box.insert(tk.END, text)

    def add_history(self, msg):
        self.history.append(msg)
        self.hist_box.delete(1.0, tk.END)
        for m in self.history[-12:]:
            self.hist_box.insert(tk.END, m + "\n")

    def update_ui(self):
        self.level_lbl.config(text=f"Level: {self.level}")
        self.score_lbl.config(text=f"Score: {self.score}")
        self.gate_lbl.config(text=f"Gates Used: {self.gates_used}")

        self.update_plots()
        self.update_state_text()
        self.root.update_idletasks()

    def update_challenge(self):
        self.challenge_box.delete(1.0, tk.END)
        self.challenge_box.insert(tk.END, CHALLENGES[self.level])

# ==============================================================
# RUN
# ==============================================================

def main():
    root = tk.Tk()
    QuantumGame2Q(root)
    root.mainloop()

if __name__ == "__main__":
    main()
