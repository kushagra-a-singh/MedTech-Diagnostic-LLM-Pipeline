import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { cn } from '@/lib/utils';
import {
  Search,
  Filter,
  Calendar,
  Eye,
  FileText,
  Download,
  MoreHorizontal,
  Clock,
  CheckCircle2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import type { HistoryItem } from '@/types';

// Mock history data
const mockHistory: HistoryItem[] = [
  {
    id: '1',
    sessionId: 'sess-001',
    studyId: 'study-001',
    patientName: 'John Smith',
    studyDate: '2024-01-15',
    modality: 'CT',
    status: 'completed',
    reportId: 'report-001',
    createdAt: '2024-01-15T10:30:00Z',
    updatedAt: '2024-01-15T11:45:00Z',
  },
  {
    id: '2',
    sessionId: 'sess-002',
    studyId: 'study-002',
    patientName: 'Jane Doe',
    studyDate: '2024-01-14',
    modality: 'MRI',
    status: 'reviewed',
    reportId: 'report-002',
    createdAt: '2024-01-14T09:15:00Z',
    updatedAt: '2024-01-14T14:30:00Z',
  },
  {
    id: '3',
    sessionId: 'sess-003',
    studyId: 'study-003',
    patientName: 'Robert Johnson',
    studyDate: '2024-01-13',
    modality: 'CT',
    status: 'in_progress',
    createdAt: '2024-01-13T16:00:00Z',
    updatedAt: '2024-01-13T16:00:00Z',
  },
  {
    id: '4',
    sessionId: 'sess-004',
    studyId: 'study-004',
    patientName: 'Emily Wilson',
    studyDate: '2024-01-12',
    modality: 'XR',
    status: 'completed',
    reportId: 'report-004',
    createdAt: '2024-01-12T08:45:00Z',
    updatedAt: '2024-01-12T10:20:00Z',
  },
  {
    id: '5',
    sessionId: 'sess-005',
    studyId: 'study-005',
    patientName: 'Michael Brown',
    studyDate: '2024-01-11',
    modality: 'CT',
    status: 'reviewed',
    reportId: 'report-005',
    createdAt: '2024-01-11T14:20:00Z',
    updatedAt: '2024-01-11T17:00:00Z',
  },
];

export default function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [modalityFilter, setModalityFilter] = useState<string>('all');
  const navigate = useNavigate();

  const filteredHistory = mockHistory.filter((item) => {
    const matchesSearch = item.patientName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.studyId.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || item.status === statusFilter;
    const matchesModality = modalityFilter === 'all' || item.modality === modalityFilter;
    return matchesSearch && matchesStatus && matchesModality;
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'in_progress':
        return (
          <Badge variant="secondary" className="gap-1">
            <Clock className="w-3 h-3" />
            In Progress
          </Badge>
        );
      case 'completed':
        return (
          <Badge variant="outline" className="gap-1 text-success border-success">
            <CheckCircle2 className="w-3 h-3" />
            Completed
          </Badge>
        );
      case 'reviewed':
        return (
          <Badge variant="outline" className="gap-1 text-primary border-primary">
            <CheckCircle2 className="w-3 h-3" />
            Reviewed
          </Badge>
        );
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-foreground">Study History</h2>
        <p className="text-muted-foreground mt-1">
          View and manage your previous imaging sessions
        </p>
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px]">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search by patient or study ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="in_progress">In Progress</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="reviewed">Reviewed</SelectItem>
            </SelectContent>
          </Select>

          <Select value={modalityFilter} onValueChange={setModalityFilter}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Modality" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Modalities</SelectItem>
              <SelectItem value="CT">CT</SelectItem>
              <SelectItem value="MRI">MRI</SelectItem>
              <SelectItem value="XR">X-Ray</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="outline">
            <Calendar className="w-4 h-4 mr-2" />
            Date Range
          </Button>
        </div>
      </Card>

      {/* Results Table */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">
              {filteredHistory.length} Results
            </CardTitle>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Patient</TableHead>
                <TableHead>Study Date</TableHead>
                <TableHead>Modality</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Last Updated</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredHistory.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                    No studies found matching your criteria
                  </TableCell>
                </TableRow>
              ) : (
                filteredHistory.map((item) => (
                  <TableRow key={item.id} className="group">
                    <TableCell>
                      <div>
                        <p className="font-medium">{item.patientName}</p>
                        <p className="text-xs text-muted-foreground">{item.studyId}</p>
                      </div>
                    </TableCell>
                    <TableCell>{item.studyDate}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{item.modality}</Badge>
                    </TableCell>
                    <TableCell>{getStatusBadge(item.status)}</TableCell>
                    <TableCell className="text-muted-foreground">
                      {new Date(item.updatedAt).toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          onClick={() => navigate(`/dashboard/view?study=${item.studyId}`)}
                          title="View in Viewer"
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                        {item.reportId && (
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            onClick={() => navigate(`/dashboard/reports?report=${item.reportId}`)}
                            title="View Report"
                          >
                            <FileText className="w-4 h-4" />
                          </Button>
                        )}
                        <Button variant="ghost" size="icon-sm">
                          <MoreHorizontal className="w-4 h-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>

          {/* Pagination */}
          <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
            <p className="text-sm text-muted-foreground">
              Showing 1-{filteredHistory.length} of {filteredHistory.length} results
            </p>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled>
                <ChevronLeft className="w-4 h-4" />
                Previous
              </Button>
              <Button variant="outline" size="sm" disabled>
                Next
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
