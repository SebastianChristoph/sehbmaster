import React, { createContext, useContext, useState, useCallback } from "react";
import { api } from "../api/client";

interface AuthContextType {
  isAdmin: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAdmin, setIsAdmin] = useState(() => !!localStorage.getItem("admin_token"));

  const login = useCallback(async (username: string, password: string) => {
    const { access_token } = await api.login(username, password);
    localStorage.setItem("admin_token", access_token);
    setIsAdmin(true);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("admin_token");
    setIsAdmin(false);
  }, []);

  return (
    <AuthContext.Provider value={{ isAdmin, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
