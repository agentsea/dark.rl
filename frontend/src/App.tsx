import { Routes, Route } from 'react-router-dom'
import LandingPage from './components/LandingPage'
import TaskPage from './components/TaskPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/tasks/:id" element={<TaskPage />} />
    </Routes>
  )
}

export default App
