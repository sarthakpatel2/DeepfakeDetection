import { useNavigate } from "react-router-dom"
import Uploader from "../components/Uploader"

export default function Home() {
	const navigate = useNavigate()

	return (
		<div className="space-y-6">
			<div className="space-y-2">
				<h1 className="text-2xl md:text-3xl font-semibold text-slate-50">Run detection</h1>
				<p className="text-slate-300 leading-relaxed">
					Upload an image or video and we’ll generate a deepfake likelihood score. For videos, the server
					processes frames in the background — results will update automatically.
				</p>
			</div>

			<div className="grid gap-4 md:grid-cols-3">
				<div className="glass-card p-4">
					<p className="text-sm font-semibold text-slate-50">1) Upload</p>
					<p className="text-sm text-slate-300 mt-1">Choose a file or drag-and-drop into the box.</p>
				</div>
				<div className="glass-card p-4">
					<p className="text-sm font-semibold text-slate-50">2) Process</p>
					<p className="text-sm text-slate-300 mt-1">We extract frames (video) and run the detector.</p>
				</div>
				<div className="glass-card p-4">
					<p className="text-sm font-semibold text-slate-50">3) Interpret</p>
					<p className="text-sm text-slate-300 mt-1">Review the score, label, and confidence hints.</p>
				</div>
			</div>

			<Uploader onUploaded={(id) => navigate(`/results/${id}`)} />
		</div>
	)
}

