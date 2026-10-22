import { ArrowLeft, RefreshCw } from "lucide-react"
import { useEffect, useState } from "react"
import { Link, useParams } from "react-router-dom"
import { getJob, getResults } from "../services/api"
import ResultCard from "../components/ResultCard"
import VideoPlayerWithFrames from "../components/VideoPlayerWithFrames"

export default function Results() {
	const { id } = useParams()
	const [job, setJob] = useState<any>(null)
	const [results, setResults] = useState<any>(null)
	const [loading, setLoading] = useState(true)
	const [error, setError] = useState<string | null>(null)

	useEffect(() => {
		let mounted = true
		async function fetchData() {
			if (!id) return
			try {
				setError(null)
				const j = await getJob(id)
				let r: any = null
				try {
					r = await getResults(id)
				} catch {
					// Results can legitimately be missing while the job is still processing.
					r = null
				}
				if (!mounted) return
				setJob(j)
				setResults(r)
				setLoading(false)
			} catch (e: any) {
				if (!mounted) return
				setError(e?.message ?? "Failed to load job")
			} finally {
				if (mounted) setLoading(false)
			}
		}
		fetchData()
		const interval = setInterval(fetchData, 2000)
		return () => {
			mounted = false
			clearInterval(interval)
		}
	}, [id])

	if (loading) {
		return (
			<div className="space-y-4">
				<div className="flex items-center justify-between">
					<h1 className="text-2xl font-semibold text-slate-50">Results</h1>
					<Link to="/home" className="btn-outline">
						<ArrowLeft className="h-4 w-4" />
						Back
					</Link>
				</div>
				<div className="glass-card p-5">
					<p className="text-slate-300">Loading job details…</p>
					<div className="mt-3 h-2 w-full rounded-full bg-slate-800 overflow-hidden">
						<div className="h-full w-1/3 bg-gradient-to-r from-sky-500 to-cyan-400 animate-pulse" />
					</div>
				</div>
			</div>
		)
	}

	if (error) {
		return (
			<div className="space-y-4">
				<div className="flex items-center justify-between">
					<h1 className="text-2xl font-semibold text-slate-50">Results</h1>
					<Link to="/home" className="btn-outline">
						<ArrowLeft className="h-4 w-4" />
						Back
					</Link>
				</div>
				<div className="glass-card p-5">
					<p className="text-rose-300 font-medium">{error}</p>
					<p className="text-slate-400 text-sm mt-1">
						If you just uploaded a video, it might still be initializing. Try refreshing in a moment.
					</p>
				</div>
			</div>
		)
	}

	if (!job) return <p className="text-rose-300">Job not found.</p>

	const statusText = typeof job?.status === "string" ? job.status : "unknown"
	const isDone = ["done", "completed", "complete", "finished", "success"].includes(
		statusText.toLowerCase()
	)
	const isError = ["error", "failed", "failure"].includes(statusText.toLowerCase())

	return (
		<div className="space-y-6">
			<div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
				<div className="space-y-1">
					<h1 className="text-2xl font-semibold text-slate-50">Results</h1>
					<p className="text-sm text-slate-400">
						Job <span className="text-slate-300 font-medium">{id}</span> •{" "}
						<span className={isError ? "text-rose-300" : isDone ? "text-emerald-300" : "text-sky-300"}>
							{statusText}
						</span>
					</p>
				</div>
				<div className="flex items-center gap-2">
					<button
						type="button"
						className="btn-outline"
						onClick={() => window.location.reload()}
						title="Refresh"
					>
						<RefreshCw className="h-4 w-4" />
						Refresh
					</button>
					<Link to="/home" className="btn-outline">
						<ArrowLeft className="h-4 w-4" />
						New upload
					</Link>
				</div>
			</div>

			{!results && !isError && (
				<div className="glass-card p-5 space-y-3">
					<p className="text-slate-50 font-semibold">Processing…</p>
					<p className="text-slate-300 text-sm leading-relaxed">
						We’re still generating results. Keep this page open — it will update automatically.
					</p>
					<div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
						<div className="h-full w-1/2 bg-gradient-to-r from-sky-500 to-cyan-400 animate-pulse" />
					</div>
				</div>
			)}

			<ResultCard result={results} job={job} />
			{job.type === "video" && job.frames && job.frames.length > 0 && (
				<VideoPlayerWithFrames frames={job.frames} />
			)}
		</div>
	)
}













