import { useState } from "react";
import GenerateButton from "./components/GenerateButton";
import SoloControls from "./components/SoloControls";
import ChordProgressionControls from "./components/ChordProgressionSettings";
import ErrorMessage from "./components/ErrorMessage";

function App() {

  const [controls, setControls] = useState({
    outputFileName: "Jazz_Solo",
    avgtempo: 120,
    style: "BEBOP",
    key: "C_MAJOR"
  });

  const [beats, setBeats] = useState([]);

  const [errorMessage, setErrorMessage] = useState(null);

  const showError = (message) => {
    if (errorMessage) return;
    setErrorMessage(message);
  };

  const dismissError = () => {
    setErrorMessage(null);
  };

  const [isGenerating, setIsGenerating] = useState(false);

  const onGenerate = async () => {
    if (isGenerating) return;
    setIsGenerating(true);

    const beatChordData = {
      beats,
      solo_info: controls
    };
    
    console.log("Sending:", JSON.stringify(beatChordData, null, 2));

    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(beatChordData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.Error || "Failed to generate solo.");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `${controls.outputFileName}.mid`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
    } catch (error) {
      showError(error.message);
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <div className="p-8 flex flex-col items-center max-w-4xl mx-auto">
      {/* TITLE */}
      <h1 className="text-5xl p-8 font-bold mb-6 text-center text-white-500 flex items-center" 
        style={{ fontFamily: "'Helvetica Neue', Arial, sans-serif"}}>
        🎷 AI JAZZ SOLO GENERATOR
      </h1>

      {/* DESCRIPTION */}
      <p className="text-lg mb-8 text-center text-gray-300">
        Generate jazz solos over your chord progressions using AI!
      </p>

      {/* CONTROLS */}

      <SoloControls controls={controls} setControls={setControls} />
      <ChordProgressionControls 
        beats={beats} 
        setBeats={setBeats} 
        onError={showError} 
      />
      <GenerateButton onGenerate={onGenerate} generating={isGenerating} />
      <ErrorMessage message={errorMessage} onDismiss={dismissError} />
    </div>
  );
}

export default App;