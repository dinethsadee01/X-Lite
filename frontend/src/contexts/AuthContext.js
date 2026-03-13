import { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

    useEffect(() => {
        // Check for existing token
        const checkAuth = async () => {
            const token = localStorage.getItem('token');
            if (token) {
                try {
                    const response = await axios.get(`${apiUrl}/auth/me`, {
                        headers: { Authorization: `Bearer ${token}` }
                    });
                    setUser(response.data);
                } catch (error) {
                    console.error('Auth check failed:', error);
                    localStorage.removeItem('token');
                }
            }
            setLoading(false);
        };
        
        checkAuth();
    }, [apiUrl]);

    const login = async (username, password) => {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);

        // OAuth2 dictates Content-Type: application/x-www-form-urlencoded
        const response = await axios.post(`${apiUrl}/auth/login`, formData, {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        
        const { access_token, user: userData } = response.data;
        localStorage.setItem('token', access_token);
        setUser(userData);
    };

    const logout = () => {
        localStorage.removeItem('token');
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, login, logout, loading }}>
            {!loading && children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    return useContext(AuthContext);
}
