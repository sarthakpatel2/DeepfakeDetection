import React from "react"
import ReactDOM from "react-dom/client"
import { createBrowserRouter, RouterProvider } from "react-router-dom"
import "./styles/index.css"
import App from "./App"
import Home from "./pages/Home"
import Landing from "./pages/Landing"
import Results from "./pages/Results"
import About from "./pages/About"

const router = createBrowserRouter([
	{
		path: "/",
		element: <App />,
		children: [
			{ index: true, element: <Landing /> },
			{ path: "/home", element: <Home /> },
			{ path: "/results/:id", element: <Results /> },
			{ path: "/about", element: <About /> },
		],
	},
])

ReactDOM.createRoot(document.getElementById("root")!).render(
	<React.StrictMode>
		<RouterProvider router={router} />
	</React.StrictMode>
)









