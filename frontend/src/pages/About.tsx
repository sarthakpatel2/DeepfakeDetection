export default function About() {
	return (
		<div className="space-y-10">
			<section className="prose">
				<h1>About this project</h1>
				<p>
					This Major Project explores <strong>deepfake detection</strong> — the task of identifying AI‑generated
					or manipulated media. Our goal is to give users an educational, hands‑on way to see how a detector
					model behaves on real images and videos.
				</p>
				<p>
					The system is built with <strong>FastAPI</strong> for the backend, <strong>MongoDB</strong> for
					persistent storage, <strong>Celery</strong> for background processing, and a lightweight{" "}
					<strong>PyTorch</strong> model for deepfake scoring. A TypeScript + React frontend ties everything
					together in a clean UI.
				</p>
				<p>
					This demo is <strong>not</strong> a production‑grade forensic tool. It is intended for learning,
					experimentation, and as a foundation that can be extended with stronger models, better datasets,
					and more advanced visualizations.
				</p>
			</section>

			<section className="space-y-4">
				<h2 className="text-xl font-semibold text-slate-50">Project Members</h2>
				<div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
					{[
						{
							name: "Aman",
							role: "API & Infrastructure",
							desc: "Implemented REST endpoints, job handling, and structured the backend so that uploads, jobs, and results flow reliably through the system.",
						},

						{
							name: "Madhav",
							role: "Backend & Model Integration",
							desc: "Integrated the PyTorch model with FastAPI, and designed the end‑to‑end inference pipeline from upload to prediction.",
						},

						{
							name: "Pragati",
							role: "Research & Coordination",
							desc: "Designed the user interface, created the landing, About, and results pages, and focused on making the detector easy and pleasant to use.",
						},


						{
							name: "Racheet",
							role: "Backend & Model Integration",
							desc: "Implemented media preprocessing, helped with job orchestration, and made sure model inputs and outputs stay consistent across services.",
						},

						{
							name: "Sarthak (GROUP LEAD)",
							role: "Architecture & Coordination",
							desc: "Coordinated the team, helped define the system architecture, documented design choices, and ensured the backend, database, and worker processes work together smoothly.",
						},

						{
							name: "Shivendra",
							role: "Frontend & Ux",
							desc: "Explored deepfake datasets and prior research, tuned model parameters, and evaluated detector performance with different settings.",
						},
						
						
					].map((m) => (
						<div key={m.name} className="member-card p-4 space-y-2">
							<p className="text-slate-50 font-semibold">{m.name}</p>
							<p className="text-sm text-slate-300">{m.role}</p>
							<p className="text-xs text-slate-400">{m.desc}</p>
						</div>
					))}
				</div>
			</section>
		</div>
	)
}













