# Simulatrix Infinity 🚀

**The Ultimate 2D Physics Engine in Python**

🌌 Simulatrix Infinity is the latest evolution of the Simulatrix series, combining cutting-edge performance, advanced collision detection, and real-time simulation capabilities. Designed for developers, educators, and hobbyists, it offers a robust framework for building interactive 2D physics simulations this engine specifically fine tuned for high frame rate future version with gui support (android) will be released soon.

---

## ⚙️ Features

* **Advanced Collision Detection**: Supports Circle-Circle, Polygon-Polygon (SAT), and Circle-Polygon (GJK + EPA) collisions.
* **Dual Integrators**: Toggle between Semi-Implicit Euler and Velocity Verlet integration methods.
* **Spatial Hashing**: Efficient broadphase collision detection for handling large numbers of entities.
* **Object Pooling**: Reduces memory allocation overhead for improved performance.
* **Real-Time Debugging**: Includes a debug HUD, grid overlay, and frame recording.
* **Interactive Playground**: GUI controls for real-time parameter adjustments and entity manipulation.
* **Cross-Platform Support**: Optimized for both desktop and mobile platforms, including Android (Pydroid).

---

## 📊 Comparison: Simulatrix Infinity vs. Previous Versions

| Feature                  | Simulatrix Infinity    | Simulatrix V3    | Simulatrix V2 | Simulatrix Original |
| ------------------------ | ---------------------- | ---------------- | ------------- | ------------------- |
| Collision Detection      | ✅ Advanced (GJK + EPA) | ✅ GJK + EPA      | ✅ SAT         | ✅ Basic             |
| Integrators              | ✅ Euler & Verlet       | ✅ Euler & Verlet | ✅ Euler       | ✅ Euler             |
| Spatial Hashing          | ✅ Optimized            | ✅ Optimized      | ✅ Basic       | ❌ Not Implemented   |
| Object Pooling           | ✅ Yes                  | ✅ Yes            | ✅ Yes         | ✅ Yes               |
| Debug HUD                | ✅ Yes                  | ✅ Yes            | ✅ Yes         | ✅ Yes               |
| GUI Controls             | ✅ Yes                 | ❌ No            | ❌ No          | ❌ No                |
| Mobile Support (Pydroid) | ✅ Yes                  | ✅ Yes            | ✅ Yes          | ✅ Yes                |
| Performance              | ⚡ High                 | ⚡ Moderate           | ⚡ High    | ⚡ High               |

---

## 🚀 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/Dhanwanth2013/Simulatrix-Infinity.git
cd Simulatrix-Infinity
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the demo**:

```bash
python main.py
```

---

## 🛠️ Usage

* **Interactive Playground**: Use the GUI controls to add/remove entities, adjust physics parameters, and observe real-time changes.
* **Customization**: Modify the `main.py` script to create custom simulations or integrate with other Python projects.
* **Documentation**: Refer to the inline comments and code documentation for detailed explanations of each component.

---

## 📈 Performance Benchmarks

Simulatrix Infinity has been optimized for performance, achieving over 60 FPS with thousands of circle entities on typical modern hardware.

**Key optimizations include:**

* **Object Pooling**: Reduces garbage collection overhead.
* **Spatial Hashing**: Minimizes collision checks.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for:

* Bug fixes
* Performance improvements
* New features or shape types
* Documentation enhancements

---

## 📄 License

This project is licensed under the MIT License. See LICENSE for details.

---

## 🙏 Acknowledgments

* Thanks to the Python, Pygame, and Numba communities for their excellent tools.
