import { Link, NavLink, Outlet } from "react-router-dom"

export default function App() {
	return (
		<div className="app-shell">
			<header className="app-header">
				<div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
					<Link to="/" className="app-header-title">
						Deepfake Detector
					</Link>
					<nav className="flex items-center gap-1">
						<NavLink
							to="/"
							end
							className={({ isActive }) =>
								`app-nav-link ${isActive ? "app-nav-link-active" : ""}`
							}
						>
							Overview
						</NavLink>
						<NavLink
							to="/home"
							className={({ isActive }) =>
								`app-nav-link ${isActive ? "app-nav-link-active" : ""}`
							}
						>
							Detect
						</NavLink>
						<NavLink
							to="/about"
							className={({ isActive }) =>
								`app-nav-link ${isActive ? "app-nav-link-active" : ""}`
							}
						>
							About
						</NavLink>
					</nav>
				</div>
			</header>
			<div className="bg-grid">
				<main className="max-w-6xl mx-auto px-4 py-10">
					<Outlet />
				</main>
				<footer className="max-w-6xl mx-auto px-4 pb-10">
					<div className="app-footer flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
						<p>
							<strong className="text-slate-300">Local demo</strong> â€” your files stay on your machine.
						</p>
						<p className="text-slate-600">
							Tip: Use a variety of real + manipulated samples for best evaluation.
						</p>
					</div>
				</footer>
			</div>
		</div>
	)
}











