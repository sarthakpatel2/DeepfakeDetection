function clamp01(v: number) {
	return Math.max(0, Math.min(1, v))
}

function formatPct(v: number) {
	return `${Math.round(v * 100)}%`
}

function confidenceFromScore(score: number) {
	// Heuristic confidence: scores near 0.5 are uncertain; further away = more confident.
	// Uncertain band: [0.45, 0.55]
	const mid = 0.5
	const band = 0.05
	const dist = Math.abs(score - mid)
	return clamp01((dist - band) / (mid - band))
}

function levelFromConfidence(c: number) {
	if (c >= 0.75) return "High"
	if (c >= 0.45) return "Medium"
	return "Low"
}

function colorFromLabel(label?: string) {
	if (label === "FAKE") return "text-rose-300"
	if (label === "REAL") return "text-emerald-300"
	if (label === "UNCERTAIN") return "text-amber-300"
	return "text-slate-300"
}

function ringFromLabel(label?: string) {
	if (label === "FAKE") return "stroke-rose-400"
	if (label === "REAL") return "stroke-emerald-400"
	if (label === "UNCERTAIN") return "stroke-amber-400"
	return "stroke-slate-400"
}

export default function ResultCard({ result, job }: { result: any; job?: any }) {
	if (!result) return null
	const { status, score, label, frames } = result
	const numericScore = typeof score === "number" ? score : null
	const confidence = numericScore == null ? null : confidenceFromScore(numericScore)
	const confidenceLevel = confidence == null ? null : levelFromConfidence(confidence)
	const mediaType = job?.type ?? (Array.isArray(frames) && frames.length > 0 ? "video" : "media")

	// Inline background color so it overrides the default card bg
	const bgStyle: React.CSSProperties =
		label === "FAKE"
			? { backgroundColor: "rgba(248, 113, 113, 0.20)" } // light red
			: label === "REAL"
				? { backgroundColor: "rgba(34, 197, 94, 0.20)" } // light green
				: {}

	const pillClass =
		label === "FAKE"
			? "pill-danger"
			: label === "REAL"
				? "pill-success"
				: label === "UNCERTAIN"
					? "pill-warning"
					: "pill-neutral"

	const meter = (() => {
		const size = 92
		const stroke = 10
		const r = (size - stroke) / 2
		const c = 2 * Math.PI * r
		const p = confidence == null ? 0 : clamp01(confidence)
		const dash = c * p
		const gap = c - dash
		return { size, stroke, r, c, dash, gap, p }
	})()

	const whyTitle =
		label === "FAKE"
			? "Why the model thinks this might be manipulated"
			: label === "REAL"
				? "Why the model thinks this looks authentic"
				: "Why the model is unsure"

	const whyBullets =
		label === "FAKE"
			? [
					"Deepfake pipelines often introduce subtle blending artifacts around facial boundaries (jawline, cheeks, hairline).",
					"Inconsistent lighting/shadows between face and background can raise suspicion.",
					"Texture irregularities (skin pores, fine details) may look overly smooth or noisy compared to real footage.",
					"For videos: temporal inconsistencies (flicker, jitter, unstable features) are a common signal.",
				]
			: label === "REAL"
				? [
						"Real media tends to have consistent facial geometry and stable fine details across the frame.",
						"Lighting and shadows often match the environment more naturally.",
						"Edges around facial features typically blend without obvious compositing artifacts.",
						"For videos: motion tends to be temporally consistent (less flicker/jitter in facial regions).",
					]
				: [
						"The score is near the uncertain region around 0.5, where the model has low separation.",
						"Low-quality inputs (compression, blur, low light) can hide artifacts or create false ones.",
						"Non-frontal faces, occlusions, or extreme expressions can reduce reliability.",
						"For videos: short clips or fast motion can make frame-level signals unstable.",
					]

	return (
		<div className="glass-card-strong p-5 space-y-5" style={bgStyle}>
			<div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
				<div className="space-y-3">
					<div className="flex items-center gap-2">
						<span className={pillClass}>{label ?? "-"}</span>
						<span className="text-xs text-slate-400">
							{mediaType}
							{mediaType === "video" && Array.isArray(job?.frames) && (
								<>
									{" "}
									• <span className="text-slate-300">{job.frames.length}</span> frames
								</>
							)}
						</span>
					</div>

					<div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
						<div>
							<p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Status</p>
							<p className="text-lg font-semibold text-slate-50">{status ?? "-"}</p>
						</div>
						<div>
							<p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
								Model score (0–1)
							</p>
							<p className="text-lg font-semibold text-sky-300">
								{numericScore != null ? numericScore.toFixed(3) : "-"}
							</p>
						</div>
						<div className="sm:col-span-1 col-span-2">
							<p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
								Confidence (heuristic)
							</p>
							<p className={`text-lg font-semibold ${colorFromLabel(label)}`}>
								{confidence == null ? "-" : `${formatPct(confidence)} • ${confidenceLevel}`}
							</p>
						</div>
					</div>

					<p className="text-sm text-slate-300 leading-relaxed">
						Higher scores mean the model leans more toward <span className="text-rose-200">FAKE</span>; lower
						scores lean toward <span className="text-emerald-200">REAL</span>. Confidence increases as the
						score moves away from the uncertain middle range.
					</p>
				</div>

				<div className="flex items-center justify-center md:justify-end">
					<div className="relative">
						<svg width={meter.size} height={meter.size} viewBox={`0 0 ${meter.size} ${meter.size}`}>
							<circle
								cx={meter.size / 2}
								cy={meter.size / 2}
								r={meter.r}
								fill="transparent"
								stroke="rgba(148, 163, 184, 0.25)"
								strokeWidth={meter.stroke}
							/>
							<circle
								cx={meter.size / 2}
								cy={meter.size / 2}
								r={meter.r}
								fill="transparent"
								className={ringFromLabel(label)}
								strokeWidth={meter.stroke}
								strokeLinecap="round"
								strokeDasharray={`${meter.dash} ${meter.gap}`}
								transform={`rotate(-90 ${meter.size / 2} ${meter.size / 2})`}
							/>
						</svg>
						<div className="absolute inset-0 flex flex-col items-center justify-center">
							<p className={`text-xl font-semibold ${colorFromLabel(label)}`}>
								{confidence == null ? "-" : formatPct(confidence)}
							</p>
							<p className="text-[11px] text-slate-400">confidence</p>
						</div>
					</div>
				</div>
			</div>

			<div className="glass-card p-4 space-y-3">
				<p className="text-sm font-semibold text-slate-50">{whyTitle}</p>
				<p className="text-sm text-slate-300 leading-relaxed">
					This demo model outputs a single score. It does <strong>not</strong> return per-feature explanations,
					so the “why” below is an educational interpretation of what detectors commonly pick up.
				</p>
				<ul className="list-disc pl-5 space-y-2 text-sm text-slate-300">
					{whyBullets.map((b) => (
						<li key={b}>{b}</li>
					))}
				</ul>

				<details className="pt-1">
					<summary className="cursor-pointer text-sm text-sky-300 font-medium">
						How the confidence is computed
					</summary>
					<p className="mt-2 text-sm text-slate-300 leading-relaxed">
						Confidence here is a <strong>frontend heuristic</strong> based on how far the score is from 0.5.
						Scores within roughly <strong>0.45–0.55</strong> are treated as low-confidence, and confidence
						increases as the score moves toward 0 or 1.
					</p>
				</details>

				<details className="pt-1">
					<summary className="cursor-pointer text-sm text-sky-300 font-medium">
						Limitations and what to check next
					</summary>
					<ul className="mt-2 list-disc pl-5 space-y-2 text-sm text-slate-300">
						<li>
							Compression, blur, low-light noise, and heavy filters can confuse detectors (false positives/negatives).
						</li>
						<li>
							If the face is small, side-profile, occluded, or fast-moving, the score is less reliable.
						</li>
						<li>
							Best practice: test multiple clips/images from the same source and look for consistent results.
						</li>
					</ul>
				</details>
			</div>
		</div>
	)
}













