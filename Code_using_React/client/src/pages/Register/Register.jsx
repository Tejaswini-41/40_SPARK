import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Register.css'; 

const Register = () => {
    const navigate = useNavigate(); 
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [success, setSuccess] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        // Basic 
        if (!name || !email || !password) {
            setError('All fields are required.');
            setLoading(false);
            return;
        }
        
        try {
            const result = await axios.post('http://localhost:3000/auth/register', { name, email, password});
            console.log(result.data);
            setSuccess(true);
            navigate('/'); //login
        } catch (err) {
            console.error(err);
            setError('Registration failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="register-container">
            <h2>Registration</h2>
            <br></br>
            <form onSubmit={handleSubmit} className="register-form">
                <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Name"
                    required
                    className="input-field"
                />
                <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Email"
                    required
                    className="input-field"
                />
                <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Password"
                    required
                    className="input-field"
                />
                <button type="submit" disabled={loading} className="register-button">
                    {loading ? 'Registering...' : 'Register'}
                </button>
            </form>

            {error && <p className="error-message">{error}</p>}
            {success && <p className="success-message">Registration successfull!</p>}

            <p>Already have an Account?</p>
            <Link to="/" className="login-link">Login</Link>
        </div>
    );
};

export default Register;