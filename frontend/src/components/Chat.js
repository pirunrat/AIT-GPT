import React, { useState } from 'react';
import './Chat.css';
import axios from 'axios';



const Chat = () => {
    const [inputWord, setInputWord] = useState('');
    const [response, setResponse] = useState('');

    const handleAPIRequest = async (e) => {
        e.preventDefault();

        try {
            
            const response = await axios.post('http://localhost:5000/question', { word: inputWord});

            if (response && response.data) {
                setResponse(response.data); 
                console.log(response.data);
            }
        } catch (error) {
            console.error('Error making request:', error);
           
        }
    };

    return (
        <div className="chat-container">
            <form onSubmit={handleAPIRequest}>
                <label>
                    Enter a word:
                    <input
                        type="text"
                        value={inputWord}
                        onChange={(e) => setInputWord(e.target.value)}
                        placeholder="Type a word..."
                    />
                </label>
                <button type="submit">Submit</button>
            </form>

            {response && (
                <div className="response-container">
                    <p>Answer:</p>
                    <p>{response}</p>
                </div>
            )}
        </div>
    );
};

export default Chat;