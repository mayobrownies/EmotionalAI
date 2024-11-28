import { useState, useEffect } from 'react';
import { marked } from 'marked';

const App = () => {
  const [message, setMessage] = useState(null);
  const [value, setValue] = useState('');
  const [prevChats, setPrevChats] = useState([]);
  const [curTitle, setCurTitle] = useState(null);

  const handleClick = (uniqueTitle) => {
    setCurTitle(uniqueTitle);
    setMessage(null);
    setValue('');
  };

  const createNewChat = () => {
    setMessage(null);
    setValue('');
    setCurTitle(null);
  };

  const getMessages = async () => {
    const options = {
      method: 'POST',
      body: JSON.stringify({
        message: value,
      }),
      headers: {
        'Content-Type': 'application/json',
      },
    };
    try {
      const response = await fetch('http://127.0.0.1:5000/ask', options);
      const data = await response.json();

      if (data.choices && data.choices.length > 0) {
        setMessage(data.choices[0].message);
      }
    } catch (error) {
      console.error(error);
    }
  };

  useEffect(() => {
    if (!curTitle && value && message) {
      setCurTitle(value);
    }
    if (curTitle && value && message) {
      setPrevChats((prevChats) => [
        ...prevChats,
        {
          title: curTitle,
          role: 'user',
          content: value,
        },
        {
          title: curTitle,
          role: message.role,
          content: message.content,
        },
      ]);
    }
  }, [message, curTitle]);

  const curChat = prevChats.filter((prevChat) => prevChat.title === curTitle);
  const uniqueTitles = Array.from(new Set(prevChats.map((prevChat) => prevChat.title)));

  return (
    <div className="app">
      <section className="side-bar">
        <button onClick={createNewChat}>New Chat</button>
        <ul className="history">
          {uniqueTitles?.map((uniqueTitle, index) => (
            <li key={index} onClick={() => handleClick(uniqueTitle)}>
              {uniqueTitle}
            </li>
          ))}
        </ul>
        <nav>
          <p></p>
        </nav>
      </section>
      <section className="main">
        <h1></h1>
        <ul className="feed">
          {curChat.map((message, index) => (
            <li key={index} className={message.role === 'user' ? 'user-message' : 'assistant-message'}>
              <p className="role">{message.role === 'user' ? 'You' : 'Assistant'}:</p>
              <p dangerouslySetInnerHTML={{ __html: marked(message.content) }} />
            </li>
          ))}
        </ul>
        <div className="bottom-section">
          <div className="input-container">
            <input value={value} onChange={(e) => setValue(e.target.value)} />
            <div id="submit" onClick={getMessages}>
              Send
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default App;
