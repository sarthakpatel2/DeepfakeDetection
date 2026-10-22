import { CheckCircle2, Shield, Sparkles, UploadCloud } from "lucide-react"
import { useNavigate } from "react-router-dom"

export default function Landing() {
	const navigate = useNavigate()

	return (
		<div className="space-y-10">
			<section className="glass-card p-7 md:p-10 overflow-hidden relative">
				<div className="absolute -top-24 -right-24 h-64 w-64 rounded-full blur-3xl bg-sky-500/10" />
				<div className="absolute -bottom-24 -left-24 h-64 w-64 rounded-full blur-3xl bg-emerald-500/10" />

				<div className="relative grid gap-10 md:grid-cols-2 md:items-center">
					<div className="space-y-5">
						<div className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-950/40 px-3 py-1 text-xs text-slate-300">
							<Sparkles className="h-4 w-4 text-sky-300" />
							Local deepfake detection demo
						</div>

						<h1 className="text-3xl md:text-5xl font-semibold leading-tight">
							<span className="app-header-title">Detect deepfakes</span> with a clean, fast workflow
						</h1>

						<p className="text-slate-300 leading-relaxed">
							Deepfakes are synthetic images and videos created with AI models that can swap faces, mimic
							voices, or recreate scenes. This app helps you upload media, run a detector, and interpret the
							results with clear explanations.
						</p>

						<div className="flex flex-wrap items-center gap-3">
							<button
								type="button"
								className="btn-primary text-base px-6 py-3"
								onClick={() => navigate("/home")}
							>
								<UploadCloud className="h-5 w-5" />
								Start detection
							</button>
							<button
								type="button"
								className="btn-outline"
								onClick={() => navigate("/about")}
							>
								<Shield className="h-4 w-4" />
								How it works
							</button>
						</div>

						<div className="grid gap-2 pt-2 text-sm text-slate-300">
							<div className="flex items-center gap-2">
								<CheckCircle2 className="h-4 w-4 text-emerald-300" />
								Image & video uploads (video is processed in the background)
							</div>
							<div className="flex items-center gap-2">
								<CheckCircle2 className="h-4 w-4 text-emerald-300" />
								Score + label with simple interpretation guidance
							</div>
							<div className="flex items-center gap-2">
								<CheckCircle2 className="h-4 w-4 text-emerald-300" />
								Runs locally — no external upload destination in this demo
							</div>
						</div>
					</div>

					<div className="space-y-4">
						<div className="glass-card-strong p-5">
							<p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
								What the score means
							</p>
							<p className="mt-2 text-slate-300 leading-relaxed">
								The model returns a score from <span className="text-sky-300 font-semibold">0.0</span> to{" "}
								<span className="text-sky-300 font-semibold">1.0</span>. Higher scores indicate a higher
								likelihood of manipulation. Scores near the middle can be uncertain.
							</p>
						</div>
						<div className="glass-card p-5">
							<p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
								Best practices
							</p>
							<ul className="mt-2 space-y-2 text-slate-300 text-sm">
								<li>Use multiple samples and compare results.</li>
								<li>Prefer clear face visibility (frontal) for consistent scoring.</li>
								<li>Interpret results as a signal — not absolute truth.</li>
							</ul>
						</div>
					</div>
				</div>
			</section>

			<section className="grid gap-4 md:grid-cols-3">
				<div className="glass-card p-5">
					<p className="text-slate-50 font-semibold">Fast upload</p>
					<p className="text-sm text-slate-300 mt-2">
						Drag and drop your file, see immediate feedback, then jump to results.
					</p>
				</div>
				<div className="glass-card p-5">
					<p className="text-slate-50 font-semibold">Clear status</p>
					<p className="text-sm text-slate-300 mt-2">
						Results update automatically while processing, especially for videos.
					</p>
				</div>
				<div className="glass-card p-5">
					<p className="text-slate-50 font-semibold">Privacy-first demo</p>
					<p className="text-sm text-slate-300 mt-2">
						This scaffold runs locally. Use non-sensitive media for demos and testing.
					</p>
				</div>
			</section>
		</div>
	)
}


