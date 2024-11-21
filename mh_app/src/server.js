import express from 'express'
import cors from 'cors'
import dotenv from'dotenv'

const PORT = 8000

const app = express()
app.use(express.json())
app.use(cors())
dotenv.config()

const OPENAI_API_KEY = process.env.OPENAI_API_KEY

console.log(OPENAI_API_KEY)

app.post('/completions', async (req, res) => {
    const options = {
        method: "POST",
        headers: {
            "Authorization": `Bearer ${OPENAI_API_KEY}`,
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            model: "gpt-4",
            messages: [{role: "user", content: req.body.message}],
            max_tokens: 100,
        })
    }
    try{
        const response = await fetch('https://api.openai.com/v1/chat/completions', options)
        const data = await response.json()
        res.send(data)
    }
    catch (error){
        console.error(error)
    }
})

app.listen(PORT, () => console.log('Your server is running on PORT ' + PORT))