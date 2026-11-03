import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import api from '../api/client';
import type { User } from '../types';

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isGuard: boolean;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  login: async () => {},
  logout: () => {},
  isLoading: true,
  isAuthenticated: false,
  isAdmin: false,
  isGuard: false,
  refreshUser: async () => {},
});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('sentinel_token'));
  const [isLoading, setIsLoading] = useState(true);

  const fetchUser = useCallback(async () => {
    if (!token) {
      setIsLoading(false);
      return;
    }
    try {
      api.setToken(token);
      const userData = await api.getMe();
      setUser(userData);
    } catch {
      setToken(null);
      setUser(null);
      api.setToken(null);
    } finally {
      setIsLoading(false);
    }
  }, [token]);

  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  const login = async (email: string, password: string) => {
    const response = await api.login(email, password);
    setToken(response.access_token);
    setUser(response.user);
    api.setToken(response.access_token);
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    api.setToken(null);
  };

  // Any API call that gets a 401 (expired/invalid JWT) dispatches this —
  // without it, isAuthenticated stays true off the last-known user object
  // while every page's own API calls silently fail, so protected routes
  // never redirect to /login and pages just render empty. Force the same
  // logout path so ProtectedRoute (in App.tsx) sends the user back to login.
  useEffect(() => {
    const handleUnauthorized = () => logout();
    window.addEventListener('sentinel:unauthorized', handleUnauthorized);
    return () => window.removeEventListener('sentinel:unauthorized', handleUnauthorized);
  }, []);

  const isAdmin = user?.role === 'admin' || user?.role === 'super_admin';
  const isGuard = user?.role === 'guard' || isAdmin;

  // Lets a page that just updated the logged-in user's own profile (e.g.
  // Face Registration saving name/phone) pull the fresh User row without
  // forcing a full re-login — reuses the same fetchUser() the initial
  // mount already relies on.
  const refreshUser = useCallback(async () => {
    await fetchUser();
  }, [fetchUser]);

  return (
    <AuthContext.Provider value={{ user, token, login, logout, isLoading, isAuthenticated: !!user, isAdmin, isGuard, refreshUser }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
