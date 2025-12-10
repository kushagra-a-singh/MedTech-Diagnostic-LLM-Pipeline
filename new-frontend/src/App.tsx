import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { SkipLink } from "@/components/ui/skip-link";
import { ProtectedRoute } from "@/components/layout/ProtectedRoute";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import UploadPage from "./pages/UploadPage";
import ViewerPage from "./pages/ViewerPage";
import ChatPage from "./pages/ChatPage";
import ReportsPage from "./pages/ReportsPage";
import HistoryPage from "./pages/HistoryPage";
import SettingsPage from "./pages/SettingsPage";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
    },
  },
});

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <SkipLink />
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            {/* Public routes */}
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
            
            {/* Protected dashboard routes */}
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <DashboardLayout />
                </ProtectedRoute>
              }
            >
              <Route index element={<Navigate to="/dashboard/upload" replace />} />
              <Route path="upload" element={
                <ErrorBoundary>
                  <UploadPage />
                </ErrorBoundary>
              } />
              <Route path="view" element={
                <ErrorBoundary>
                  <ViewerPage />
                </ErrorBoundary>
              } />
              <Route path="chat" element={
                <ErrorBoundary>
                  <ChatPage />
                </ErrorBoundary>
              } />
              <Route path="reports" element={
                <ErrorBoundary>
                  <ReportsPage />
                </ErrorBoundary>
              } />
              <Route path="history" element={
                <ErrorBoundary>
                  <HistoryPage />
                </ErrorBoundary>
              } />
              <Route path="settings" element={
                <ErrorBoundary>
                  <SettingsPage />
                </ErrorBoundary>
              } />
            </Route>
            
            {/* Redirect root to login or dashboard */}
            <Route path="/" element={<Navigate to="/login" replace />} />
            
            {/* 404 */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;
