import { FileVideo2, Image as ImageIcon, UploadCloud, X } from "lucide-react"
import { useMemo, useRef, useState } from "react"
import { uploadFile } from "../services/api"

export default function Uploader({ onUploaded }: { onUploaded: (id: string) => void }) {
	const [busy, setBusy] = useState(false)
	const [error, setError] = useState<string | null>(null)
	const [file, setFile] = useState<File | null>(null)
	const inputRef = useRef<HTMLInputElement | null>(null)

	const fileMeta = useMemo(() => {
		if (!file) return null
		const mb = file.size / (1024 * 1024)
		return {
			name: file.name,
			type: file.type || "unknown",
			sizeLabel: mb < 1 ? `${Math.round(file.size / 1024)} KB` : `${mb.toFixed(1)} MB`,
			isVideo: file.type.startsWith("video/"),
			isImage: file.type.startsWith("image/"),
		}
	}, [file])

	function validateSelected(f: File) {
		// Keep this permissive; backend decides what it can handle.
		// We only guard common mistakes and huge files.
		const maxMB = 250
		if (f.size > maxMB * 1024 * 1024) {
			throw new Error(`File is too large. Max allowed size is ${maxMB} MB.`)
		}
		if (!(f.type.startsWith("image/") || f.type.startsWith("video/"))) {
			throw new Error("Unsupported file type. Please upload an image or a video.")
		}
	}

	async function submitSelected(f: File) {
		setBusy(true)
		setError(null)
		try {
			const res = await uploadFile(f)
			onUploaded(res.job_id)
		} catch (err: any) {
			setError(err?.message ?? "Upload failed")
		} finally {
			setBusy(false)
		}
	}

	async function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
		if (!e.target.files || e.target.files.length === 0) return
		const f = e.target.files[0]
		try {
			validateSelected(f)
			setFile(f)
			await submitSelected(f)
		} catch (err: any) {
			setError(err?.message ?? "Upload failed")
		} finally {
			e.target.value = ""
		}
	}

	async function onDrop(e: React.DragEvent<HTMLLabelElement>) {
		e.preventDefault()
		if (busy) return
		const dropped = e.dataTransfer.files?.[0]
		if (!dropped) return
		try {
			validateSelected(dropped)
			setFile(dropped)
			await submitSelected(dropped)
		} catch (err: any) {
			setError(err?.message ?? "Upload failed")
		}
	}

	function clearSelected() {
		setFile(null)
		setError(null)
		if (inputRef.current) inputRef.current.value = ""
	}

	return (
		<div className="space-y-4 glass-card p-5">
			<div className="space-y-1">
				<h2 className="text-xl font-semibold text-slate-50">Upload media</h2>
				<p className="upload-hint">
					Drag and drop, or choose a file. Video may take longer while frames are extracted.
				</p>
			</div>

			<label
				className={`upload-box cursor-pointer ${busy ? "opacity-70 cursor-not-allowed" : ""}`}
				onDragOver={(e) => e.preventDefault()}
				onDrop={onDrop}
			>
				<div className="flex items-start gap-3 w-full">
					<div className="mt-0.5 rounded-xl border border-slate-800 bg-slate-950/40 p-2.5">
						<UploadCloud className="h-5 w-5 text-sky-300" />
					</div>
					<div className="space-y-1 flex-1">
						<p className="text-sm font-medium text-slate-100">
							Drop an image/video here, or <span className="text-sky-300">browse</span>
						</p>
						<p className="upload-hint">
							Supported: <strong>image/*</strong>, <strong>video/*</strong>. Files stay local in this demo.
						</p>
					</div>
					{fileMeta && !busy && (
						<button type="button" className="btn-outline" onClick={clearSelected} title="Clear file">
							<X className="h-4 w-4" />
							Clear
						</button>
					)}
				</div>
				<input
					type="file"
					ref={inputRef}
					onChange={handleChange}
					disabled={busy}
					className="hidden"
					accept="image/*,video/*"
				/>
			</label>

			{fileMeta && (
				<div className="glass-card-strong p-4">
					<div className="flex items-center justify-between gap-3">
						<div className="flex items-center gap-3 min-w-0">
							<div className="rounded-xl border border-slate-800 bg-slate-950/40 p-2">
								{fileMeta.isVideo ? (
									<FileVideo2 className="h-5 w-5 text-cyan-300" />
								) : (
									<ImageIcon className="h-5 w-5 text-emerald-300" />
								)}
							</div>
							<div className="min-w-0">
								<p className="text-sm font-semibold text-slate-50 truncate">{fileMeta.name}</p>
								<p className="text-xs text-slate-400 truncate">
									{fileMeta.type} • {fileMeta.sizeLabel}
								</p>
							</div>
						</div>
						{busy && <p className="upload-status">Uploading…</p>}
					</div>
				</div>
			)}

			<div className="flex items-center justify-between">
				<p className="upload-hint">
					For best results: ensure the face is visible and avoid extreme motion blur.
				</p>
				{!busy && (
					<p className="text-xs text-slate-500">
						Max recommended size: <strong>250 MB</strong>
					</p>
				)}
			</div>

			{error && <p className="upload-error">{error}</p>}
		</div>
	)
}







